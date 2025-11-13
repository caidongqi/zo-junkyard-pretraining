import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Iterable

import torch


class CheckpointManager:
    """
    Manage latest and milestone checkpoints during training.
    - Always keeps an up-to-date `latest` checkpoint (overwritten each save).
    - Persists milestone checkpoints in `snapshots/` only when the training loss
      is lower than the global best loss by at least `snapshot_delta`.
    - This ensures snapshots are saved only when achieving new milestones,
      preventing excessive snapshot creation.
    """

    def __init__(
        self,
        base_dir: Optional[str],
        logger=None,
        snapshot_delta: float = 0.5,
        snapshot_delta_2: float = 0.1,
        turning_point: float = 4.5
    ):
        self.logger = logger
        self.snapshot_delta = snapshot_delta
        self.snapshot_delta_2 = snapshot_delta_2
        self.turning_point = turning_point
        self.base_dir = Path(base_dir) if base_dir else None
        self.best_loss: Optional[float] = None  # Track the global best (lowest) loss

        if self.base_dir:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            (self.base_dir / "snapshots").mkdir(exist_ok=True)

    # --------------------------------------------------------------------- #
    # Helpers
    def _log(self, message: str, level: str = "info"):
        if self.logger:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(message)
        else:
            print(message)

    def _write_checkpoint(
        self,
        destination: Path,
        model,
        tokenizer,
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Path:
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(destination)
        if tokenizer is not None:
            tokenizer_dir = destination / "tokenizer"
            tokenizer.save_pretrained(tokenizer_dir)

        if optimizer_state is not None:
            torch.save(optimizer_state, destination / "optimizer.pt")

        metadata_path = destination / "training_state.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return destination

    # --------------------------------------------------------------------- #
    # Public API
    def initialize_from_metadata(self, resume_metadata: Optional[Dict[str, Any]]):
        if resume_metadata is None:
            return
        best_loss = resume_metadata.get("best_loss")
        if isinstance(best_loss, (int, float)):
            self.best_loss = float(best_loss)

    def save_latest(
        self,
        model,
        tokenizer,
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Optional[Path]:
        if self.base_dir is None:
            return None

        metadata = dict(metadata)  # copy
        metadata["checkpoint_type"] = metadata.get("checkpoint_type", "latest")
        metadata["best_loss"] = self.best_loss

        latest_dir = self.base_dir / "latest"
        saved_path = self._write_checkpoint(latest_dir, model, tokenizer, optimizer_state, metadata)
        self._log(f"Latest checkpoint saved to {saved_path}")
        return saved_path

    def maybe_save_snapshot(
        self,
        loss_value: float,
        model,
        tokenizer,
        optimizer_state: Optional[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> Optional[Path]:
        if self.base_dir is None:
            return None

        # Only save if this is the first snapshot OR if loss is lower than best_loss by snapshot_delta
        should_save = False
        if self.best_loss is None:
            # First snapshot
            should_save = True
        else:
            # Use different snapshot delta based on turning point
            if loss_value > self.turning_point:
                delta = self.snapshot_delta_2
            else:
                delta = self.snapshot_delta
            if loss_value <= self.best_loss - delta:
                # Loss has decreased by at least the applicable snapshot_delta from the global best
                should_save = True

        if not should_save:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"step{metadata.get('step', 'na')}_loss{loss_value:.4f}_{timestamp}"
        snapshot_dir = self.base_dir / "snapshots" / snapshot_name

        metadata = dict(metadata)
        metadata["checkpoint_type"] = "snapshot"
        metadata["snapshot_delta"] = self.snapshot_delta
        metadata["best_loss"] = loss_value

        saved_path = self._write_checkpoint(
            snapshot_dir, model, tokenizer, optimizer_state, metadata
        )
        self.best_loss = loss_value
        self._log(f"Snapshot checkpoint saved to {saved_path} (new best loss: {loss_value:.4f})")
        return saved_path


class EvaluationManager:
    """
    Run evaluation on downstream datasets and log results.
    Stores JSONL records in the configured output file.
    
    Now includes evaluation for:
    - SST-2 (Perplexity)
    - SQuAD (Perplexity)
    - LAMBADA (Perplexity and Accuracy)
    """

    def __init__(
        self,
        device: str,
        results_file: Path,
        logger=None,
        max_samples: int = 128,
        block_size: int = 256,
    ):
        self.device = device
        self.results_file = Path(results_file)
        self.logger = logger
        self.max_samples = max_samples
        self.block_size = block_size
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.results_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            from datasets import load_dataset  # noqa: F401
            self.datasets_available = True
        except Exception as err:  # pragma: no cover - defensive
            self.datasets_available = False
            self._log(f"Datasets library unavailable: {err}. Evaluation disabled.", level="warning")

    # ------------------------------------------------------------------ #
    def _log(self, message: str, level: str = "info"):
        if self.logger:
            log_fn = getattr(self.logger, level, self.logger.info)
            log_fn(message)
        else:
            print(message)

    def _prepare_texts(self, dataset_id: str) -> Iterable[str]:
        from datasets import load_dataset

        # 优先使用本地缓存，如果不存在才下载
        download_mode = "reuse_cache_if_exists"

        if dataset_id == "sst2":
            dataset = load_dataset("glue", "sst2", split="validation", download_mode=download_mode)
            for example in dataset.select(range(min(self.max_samples, len(dataset)))):
                yield example["sentence"]
        elif dataset_id == "squad":
            dataset = load_dataset("squad", split="validation", download_mode=download_mode)
            count = 0
            for example in dataset:
                answers = example.get("answers", {}).get("text", [])
                answer_text = answers[0] if answers else ""
                text = " ".join(
                    filter(
                        None,
                        [example.get("context"), example.get("question"), answer_text],
                    )
                )
                if text:
                    yield text
                    count += 1
                    if count >= self.max_samples:
                        break
        # [新增] 为 LAMBADA 数据集做准备
        elif dataset_id == "lambada":
            # LAMBADA 评估通常在 test split 上进行
            dataset = load_dataset("lambada", split="test", download_mode=download_mode)
            for example in dataset.select(range(min(self.max_samples, len(dataset)))):
                yield example["text"]
        else:
            raise ValueError(f"Unknown dataset id for evaluation: {dataset_id}")

    def _compute_perplexity(self, model, tokenizer, texts: Iterable[str]) -> float:
        model_was_training = model.training
        model.eval()

        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            if not text:
                continue
            encoded = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.block_size,
            )
            input_ids = encoded["input_ids"].to(self.device)
            labels = input_ids.clone()

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = self.loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            token_count = shift_labels.numel()
            total_loss += loss.item() * token_count
            total_tokens += token_count

        if model_was_training:
            model.train()

        if total_tokens == 0:
            return float("inf")
        return math.exp(total_loss / total_tokens)
    
    # [新增] 为 LAMBADA 计算准确率的方法
    def _compute_accuracy_lambada(self, model, tokenizer, texts: Iterable[str]) -> float:
        model_was_training = model.training
        model.eval()
        
        correct_predictions = 0
        total_samples = 0

        for text in texts:
            text = text.strip()
            if not text or ' ' not in text:
                continue

            # 分割上下文和最后一个词
            try:
                context, target_word = text.rsplit(' ', 1)
            except ValueError:
                continue
            
            total_samples += 1
            
            inputs = tokenizer(context, return_tensors="pt", truncation=True, max_length=self.block_size - 1).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # 获取对下一个词的预测 logits
                last_token_logits = outputs.logits[0, -1, :]
            
            # 找到概率最高的 token
            predicted_token_id = torch.argmax(last_token_logits).item()
            
            # 正确解码：将预测的token添加到上下文中，然后解码整个序列
            # 这样可以处理subword tokenization的情况
            context_token_ids = inputs['input_ids'][0].tolist()
            full_sequence = context_token_ids + [predicted_token_id]
            
            # 解码完整序列
            decoded_text = tokenizer.decode(full_sequence, skip_special_tokens=True)
            
            # 提取新生成的部分（即预测的词）
            # 解码后的文本应该包含原始上下文加上新token
            # 我们需要找到新生成的部分
            context_decoded = tokenizer.decode(context_token_ids, skip_special_tokens=True)
            
            # 如果解码后的文本以原始上下文开头，提取剩余部分作为预测词
            if decoded_text.startswith(context_decoded):
                predicted_word = decoded_text[len(context_decoded):].strip()
                # 如果预测词包含空格，只取第一个词
                if ' ' in predicted_word:
                    predicted_word = predicted_word.split()[0]
            else:
                # 如果格式不匹配，尝试直接解码单个token
                predicted_word = tokenizer.decode([predicted_token_id], skip_special_tokens=True).strip()
            
            # 比较预测词和目标词（去除标点符号和空格）
            predicted_word_clean = predicted_word.lower().strip('.,!?;:')
            target_word_clean = target_word.lower().strip('.,!?;:')
            
            if predicted_word_clean == target_word_clean:
                correct_predictions += 1
        
        if model_was_training:
            model.train()
            
        if total_samples == 0:
            return 0.0
        
        return correct_predictions / total_samples

    def evaluate(
        self,
        model,
        tokenizer,
        *,
        step: int,
        epoch: int,
        train_loss: float,
        checkpoint_path: Optional[str],
        checkpoint_type: str,
    ) -> Optional[Dict[str, Any]]:
        if not self.datasets_available:
            return None

        metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "train_loss": train_loss,
            "checkpoint_path": checkpoint_path,
            "checkpoint_type": checkpoint_type,
        }

        try:
            metrics["sst2_perplexity"] = self._compute_perplexity(
                model, tokenizer, self._prepare_texts("sst2")
            )
        except Exception as err:
            metrics["sst2_error"] = str(err)
            self._log(f"SST-2 evaluation failed: {err}", level="warning")

        try:
            metrics["squad_perplexity"] = self._compute_perplexity(
                model, tokenizer, self._prepare_texts("squad")
            )
        except Exception as err:
            metrics["squad_error"] = str(err)
            self._log(f"SQuAD evaluation failed: {err}", level="warning")
            
        # [新增] 对 LAMBADA 进行评估
        try:
            lambada_texts = list(self._prepare_texts("lambada"))
            metrics["lambada_perplexity"] = self._compute_perplexity(
                model, tokenizer, lambada_texts
            )
            metrics["lambada_accuracy"] = self._compute_accuracy_lambada(
                model, tokenizer, lambada_texts
            )
        except Exception as err:
            metrics["lambada_error"] = str(err)
            self._log(f"LAMBADA evaluation failed: {err}", level="warning")

        with self.results_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        # 更新日志输出以包含 LAMBADA 指标
        def format_metric(value):
            return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)

        log_message = (
            f"Evaluation recorded at step {step}: "
            f"SST-2 ppl={format_metric(metrics.get('sst2_perplexity', 'N/A'))}, "
            f"SQuAD ppl={format_metric(metrics.get('squad_perplexity', 'N/A'))}, "
            f"LAMBADA ppl={format_metric(metrics.get('lambada_perplexity', 'N/A'))}, "
            f"LAMBADA acc={format_metric(metrics.get('lambada_accuracy', 'N/A'))}"
        )
        self._log(log_message)

        return metrics