# -*- coding: utf-8 -*-
"""
step1_realtime_transcribe.py
---------------------------------
Whisper large-v2 を用いてオフライン音声ファイル (WAV) を擬似リアルタイムに逐次文字起こしするデモスクリプト。
realtime-whisper のアイデアを参考に、音声をチャンクに分割し、
チャンクを追加しながら毎回累積オーディオに対して再度推論を行い、
より精度の高いテキストを得る。

使い方:
  python step1_realtime_transcribe.py --file monologue_sample_20250516.wav --chunk_sec 5 --out result_step1.txt

引数:
  --file       入力音声(モノラル WAV / 任意サンプリングレート)。
  --chunk_sec  何秒ごとに逐次推論を行うか。デフォルト5秒。
  --model      Whisperモデル名 (デフォルト "large-v2")。
  --language   音声言語 (デフォルト "ja")。
  --out        出力テキストファイル。指定がない場合は"transcription_step1.txt"に保存。

注意:
  ・CUDA/GPU が利用可能な環境での実行を推奨します。
  ・メモリ節約のため、最後の max_buffer_sec (デフォルト 30 秒) だけを保持しています。
  ・実際のマイク入力によるリアルタイム処理は sounddevice 等で容易に拡張できます。
"""

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import whisper
import importlib

# --- Optional dependency (wrapped in try/except) ---
try:
    import librosa  # type: ignore  # for resampling
except ModuleNotFoundError:
    librosa = None


def load_audio_mono(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file as mono waveform with the target sample rate.

    1. まず soundfile でロード（高速）。
    2. Resample が必要な場合、librosa が存在すればそれを使用。
       librosa がインストールされていない場合は whisper 内蔵の
       `whisper.audio.load_audio` ＋ numpy 配列変換でフォールバックします。
    """
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    if sr == target_sr:
        return audio

    # resample
    if librosa is not None:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        return audio
    else:
        # whisper のユーティリティを使用 (ffmpeg 依存)
        print("[WARN] librosa が見つからないため、whisper.audio.load_audio でリサンプリングします。")
        from whisper.audio import load_audio
        return load_audio(str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, required=True, help="Input WAV file (mono or stereo)")
    parser.add_argument("--chunk_sec", type=float, default=5.0, help="Seconds per transcription chunk")
    parser.add_argument("--model", type=str, default="large-v2", help="Whisper model name")
    parser.add_argument("--language", type=str, default="ja", help="Language code")
    parser.add_argument("--out", type=Path, default=Path("transcription_step1.txt"))
    parser.add_argument("--max_buffer_sec", type=float, default=30.0, help="Max seconds to keep in rolling buffer")

    args = parser.parse_args()

    print(f"[INFO] モデル '{args.model}' をロードしています ...")
    model = whisper.load_model(args.model)

    print(f"[INFO] オーディオ '{args.file}' を読み込み中 ...")
    waveform = load_audio_mono(args.file, target_sr=16000)
    total_samples = len(waveform)
    chunk_samples = int(args.chunk_sec * 16000)
    max_buffer_samples = int(args.max_buffer_sec * 16000)

    transcripts = []
    rolling_buffer: list[np.ndarray] = []

    for offset in range(0, total_samples, chunk_samples):
        chunk = waveform[offset: offset + chunk_samples]
        rolling_buffer.append(chunk)
        # バッファが一定長を超えないように古い部分を削除
        current_buffer = np.concatenate(rolling_buffer)
        if len(current_buffer) > max_buffer_samples:
            # truncate from the front
            excess = len(current_buffer) - max_buffer_samples
            current_buffer = current_buffer[excess:]
            # 同期して rolling_buffer を更新
            # 再分割して rolling_buffer へ
            rolling_buffer = [current_buffer]

        print("[INFO] 推論実行中 ... (進捗 {:.1f}% )".format(100 * offset / total_samples))
        start_time = time.time()
        result = model.transcribe(current_buffer, language=args.language, fp16=torch.cuda.is_available())
        elapsed = time.time() - start_time
        print(f"[INFO] 推論完了: {elapsed:.2f}s | テキスト: {result['text']}")

        transcripts.append(result["text"].strip())

    # 最終結果を保存
    final_text = "\n".join(transcripts)
    args.out.write_text(final_text, encoding="utf-8")
    print(f"[INFO] 字幕を '{args.out}' に保存しました。")


if __name__ == "__main__":
    main() 