# realtime-whisper

## Summary
A server that provides real-time (streaming) transcription using Whisper in Japanese.

Authentication is possible using the connection URL and API key.

Real-time CER is around 3%, and CER after follow-up processing is below 2%.

The server is designed for NVIDIA A100 and can handle 5 audio streams simultaneously. Delays occur when more than 5 audio streams are input.

The server cost is approximately 10 yen per hour at Japanese electricity rates. (estimate based on A100)