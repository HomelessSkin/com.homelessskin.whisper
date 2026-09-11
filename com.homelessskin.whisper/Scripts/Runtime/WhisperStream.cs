using System;
using System.Collections.Generic;
using System.Threading.Tasks;

using Core;

using Whisper.Utils;
// ReSharper disable once RedundantUsingDirective

namespace Whisper
{
    public delegate void OnStreamSegmentFinishedDelegate(WhisperResult segment);

    /// <summary>
    /// Parameters of whisper streaming processing.
    /// </summary>
    public class WhisperStreamParams
    {
        /// <summary>
        /// Regular whisper inference params.
        /// </summary>
        public readonly WhisperParams InferenceParam;

        /// <summary>
        /// Audio stream frequency. Can't change during transcription.
        /// </summary>
        public readonly int Frequency;
        /// <summary>
        /// Audio stream channels count. Can't change during transcription.
        /// </summary>
        public readonly int Channels;
        /// <summary>
        /// Minimal portions of audio that will be processed by whisper stream in seconds.
        /// </summary>
        public readonly float StepSec;
        /// <summary>
        /// Minimal portions of audio that will be processed by whisper in audio samples.    
        /// </summary>
        public readonly int StepSamples;
        /// <summary>
        /// How many seconds of previous segment will be used for current segment.
        /// </summary>
        public readonly float KeepSec;
        /// <summary>
        /// How many samples of previous audio chunk will be used for current chunk.
        /// </summary>
        public readonly int KeepSamples;
        /// <summary>
        /// How many seconds of audio will be recurrently transcribe until context update.
        /// </summary>
        public readonly float LengthSec;
        /// <summary>
        /// How many samples of audio will be recurrently transcribe until context update.
        /// </summary>
        public readonly int LengthSamples;
        /// <summary>
        /// How many recurrent iterations will be used for one chunk?
        /// </summary>
        public readonly int StepsCount;

        public WhisperStreamParams(WhisperParams inferenceParam,
            int frequency, int channels,
            float stepSec = 3f, float keepSec = 0.2f, float lengthSec = 10f)
        {
            InferenceParam = inferenceParam;
            Frequency = frequency;
            Channels = channels;

            StepSec = stepSec;
            StepSamples = (int)(StepSec * Frequency * Channels);

            KeepSec = keepSec;
            KeepSamples = (int)(KeepSec * frequency * channels);

            LengthSec = lengthSec;
            LengthSamples = (int)(LengthSec * frequency * channels);

            StepsCount = Math.Max(1, (int)(LengthSec / StepSec) - 1);
        }
    }

    /// <summary>
    /// Handling all streaming logic (sliding-window, VAD, etc).
    /// </summary>
    public class WhisperStream
    {
        /// <summary>
        /// Raised when whisper finished current segment transcript. 
        /// </summary>
        public event OnStreamSegmentFinishedDelegate OnSegmentFinished;

        readonly WhisperWrapper _wrapper;
        readonly WhisperStreamParams _param;
        readonly MicrophoneRecord _microphone;

        bool _isStreaming;

        readonly List<float> _newBuffer = new List<float>();

        Task<WhisperResult> _task;

        /// <summary>
        /// Create a new instance of Whisper streaming transcription.
        /// </summary>
        /// <param name="wrapper">Loaded Whisper model which will be used for transcription.</param>
        /// <param name="param">Whisper streaming parameters.</param>
        /// <param name="microphone">Optional microphone input for stream.</param>
        public WhisperStream(WhisperWrapper wrapper, WhisperStreamParams param, MicrophoneRecord microphone = null)
        {
            _wrapper = wrapper;
            _param = param;
            _microphone = microphone;
        }

        /// <summary>
        /// Manually add a new chunk of audio to streaming.
        /// Make sure to call <see cref="StartStream"/> first.
        /// </summary>
        /// <remarks>
        /// If you set microphone into constructor, it will be called automatically.
        /// </remarks>
        public void AddToStream(AudioChunk chunk)
        {
            if (!_isStreaming)
            {
                Log.Warning(this, "Start streaming first!");

                return;
            }

            _newBuffer.AddRange(chunk.Data);
        }
        /// <summary>
        /// Stop current streaming transcription. It will process last
        /// audio chunks and raise <see cref="OnStreamFinished"/> when it's done.
        /// </summary>
        public async void StopStream()
        {
            if (!_isStreaming)
            {
                Log.Warning(this, "Start streaming first!");

                return;
            }

            _isStreaming = false;

            // unsubscribe from microphone events for now
            if (_microphone != null)
            {
                _microphone.OnChunkReady -= MicrophoneOnChunkReady;
                _microphone.OnRecordStop -= MicrophoneOnRecordStop;
            }

            // first wait until last task complete
            if (_task != null)
                await _task;

            // reset stream and drop audio buffer
            Reset();
        }
        /// <summary>
        /// Start a new streaming transcription. Must be called before
        /// you start adding new audio chunks.
        /// </summary>
        /// <remarks>
        /// If you set microphone into constructor, it will start listening to it.
        /// Make sure you started microphone by <see cref="MicrophoneRecord.StartRecord"/>.
        /// There is no need to add audio chunks manually using <see cref="AddToStream"/>.
        /// </remarks>
        public void StartStream()
        {
            if (_isStreaming)
            {
                Log.Warning(this, "Stream is already working!");

                return;
            }

            _isStreaming = true;

            // if we set microphone - streaming works in auto mode
            if (_microphone != null)
            {
                _microphone.OnChunkReady += MicrophoneOnChunkReady;
                _microphone.OnRecordStop += MicrophoneOnRecordStop;
            }
        }

        void Reset()
        {
            _newBuffer.Clear();
        }
        void MicrophoneOnChunkReady(AudioChunk chunk)
        {
            AddToStream(chunk);
        }
        void MicrophoneOnRecordStop(AudioChunk recordedAudio)
        {
            StopStream();
        }
    }
}