"""CLAP으로 오디오와 영어 검색문을 같은 공간의 벡터로 변환"""

from collections.abc import Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import ClapModel, ClapProcessor


class ClapEncoder:
    def __init__(
        self,
        model_id: str = "laion/clap-htsat-unfused",
        *,
        device: str = "cpu",
        revision: str = "main",
        local_files_only: bool = False,
    ):
        # 모델과 전처리기는 같은 체크포인트를 사용
        # 처음 실행하면 캐시에 없는 모델 파일을 다운로드
        self.processor = ClapProcessor.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        )

        self.device = torch.device(device)

        self.model = ClapModel.from_pretrained(
            model_id,
            revision=revision,
            local_files_only=local_files_only,
        ).to(self.device)

        # 드롭아웃 등 학습용 동작을 끔
        self.model.eval()

        extractor = self.processor.feature_extractor

        self.sample_rate = int(extractor.sampling_rate)
        self.max_samples = round(
            self.sample_rate * extractor.max_length_s
        )
        self.dimension = int(self.model.config.projection_dim)

        # CLAP의 RoBERTa 위치 임베딩에 필요한 여유를 제외
        self.max_text_tokens = min(
            int(self.processor.tokenizer.model_max_length),
            int(self.model.config.text_config.max_position_embeddings) - 2,
        )

    @staticmethod
    def _check_batch_size(batch_size: int) -> None:
        """배치 크기가 양의 정수인지 확인"""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("배치 크기는 정수여야 합니다.")

        if batch_size <= 0:
            raise ValueError("배치 크기는 0보다 커야 합니다.")

    def _vectors(
        self,
        output: object,
        expected_rows: int,
    ) -> NDArray[np.float32]:
        """모델 반환값을 검증하고 정규화한 NumPy 배열로 변환"""
        # 현재 Transformers 5에서는 투영된 벡터가
        # 반환 객체의 pooler_output에 들어있다.
        features = (
            output
            if isinstance(output, torch.Tensor)
            else getattr(output, "pooler_output", None)
        )

        if not isinstance(features, torch.Tensor):
            raise RuntimeError("CLAP이 임베딩 텐서를 반환하지 않았습니다.")

        if tuple(features.shape) != (
            expected_rows,
            self.dimension,
        ):
            raise RuntimeError("CLAP 임베딩의 크기가 예상과 다릅니다.")

        features = features.float()

        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("임베딩에 NaN 또는 무한대 값이 있습니다.")

        norms = torch.linalg.vector_norm(
            features,
            dim=1,
            keepdim=True,
        )

        if bool((norms <= 1e-12).any()):
            raise RuntimeError("정규화할 수 없는 임베딩이 생성됐습니다.")

        # 벡터 길이를 1로 맞춤
        # 이렇게 하면 두 벡터의 내적이 코사인 유사도가 됨
        return (
            (features / norms)
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

    @torch.inference_mode()
    def encode_audio(
        self,
        waveforms: Sequence[NDArray[np.float32]],
        *,
        sample_rate: int,
        batch_size: int = 4,
    ) -> NDArray[np.float32]:
        """모노 구간 목록을 받아 입력 순서대로 임베딩을 반환"""
        self._check_batch_size(batch_size)

        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("샘플레이트는 정수여야 합니다.")

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"오디오를 {self.sample_rate}Hz로 전처리해야 합니다."
            )

        if len(waveforms) == 0:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        results = []

        for start in range(0, len(waveforms), batch_size):
            batch = []

            for waveform in waveforms[start:start + batch_size]:
                # 읽기 전용 구간 배열을 모델 입력용 복사본으로 바꾸기
                samples = np.array(
                    waveform,
                    dtype=np.float32,
                    copy=True,
                )

                if (
                    samples.ndim != 1
                    or not 0 < samples.size <= self.max_samples
                ):
                    raise ValueError(
                        "오디오는 모델 길이 제한 안의 "
                        "비어 있지 않은 모노 구간이어야 합니다."
                    )

                if not np.isfinite(samples).all():
                    raise ValueError(
                        "오디오에 NaN 또는 무한대 값이 있습니다."
                    )

                batch.append(samples)

            # 짧은 입력의 길이는 체크포인트의 repeatpad 방식으로 맞춤
            # 긴 입력은 위에서 거부하여 무작위 잘림을 방지
            inputs = self.processor.feature_extractor(
                batch,
                sampling_rate=sample_rate,
                padding="repeatpad",
                truncation="rand_trunc",
                return_tensors="pt",
            ).to(self.device)

            output = self.model.get_audio_features(
                **inputs,
                return_dict=True,
            )

            results.append(
                self._vectors(output, len(batch))
            )

        return np.concatenate(results, axis=0)

    @torch.inference_mode()
    def encode_text(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 16,
    ) -> NDArray[np.float32]:
        """영어 검색문 목록을 같은 공간의 임베딩으로 변환."""
        self._check_batch_size(batch_size)

        if isinstance(texts, str):
            raise TypeError("검색문은 문자열 목록으로 전달하세요.")

        if any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "검색문은 비어 있지 않은 문자열이어야 합니다."
            )

        if len(texts) == 0:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        results = []

        for start in range(0, len(texts), batch_size):
            batch = [
                text.strip()
                for text in texts[start:start + batch_size]
            ]

            inputs = self.processor.tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            # 긴 검색문을 몰래 자르면 의미가 바뀔 수 있으므로 거부
            if inputs["input_ids"].shape[1] > self.max_text_tokens:
                raise ValueError(
                    f"검색문을 {self.max_text_tokens}토큰 이하로 줄여주세요."
                )

            output = self.model.get_text_features(
                **inputs.to(self.device),
                return_dict=True,
            )

            results.append(
                self._vectors(output, len(batch))
            )

        return np.concatenate(results, axis=0)