using System;
using System.IO;
using System.Threading.Tasks;

using Core;

using Unity.Collections;

using UnityEngine;

using Whisper.Native;
using Whisper.Utils;

namespace Whisper
{
    /// <summary>
    /// Manages Whisper model lifecycle in Unity scene.
    /// </summary>
    public class WhisperManager : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] bool initOnAwake = true;
        [SerializeField] bool isModelPathInStreamingAssets = true;
        [SerializeField] string modelPath = "Whisper/ggml-tiny.bin";

        [Header("Inference")]
        [SerializeField] bool useGpu;
        [SerializeField] bool flashAttention;

        [Header("Language")]
        [SerializeField] bool translateToEnglish;
        [SerializeField] string language = "en";

        [Header("Advanced settings")]
        [SerializeField] WhisperSamplingStrategy strategy = WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY;

        [Header("Streaming settings")]
        [SerializeField] float stepSec = 3f;
        [SerializeField] float keepSec = 0.2f;
        [SerializeField] float lengthSec = 10f;

        /// <summary>
        /// Raised when whisper transcribed a new text segment from audio. 
        /// </summary>
        public event OnNewSegmentDelegate OnNewSegment;
        /// <summary>
        /// Raised when whisper made some progress in transcribing audio.
        /// Progress changes from 0 to 100 included.
        /// </summary>
        public event OnProgressDelegate OnProgress;

        WhisperWrapper _whisper;
        WhisperParams _params;

        readonly MainThreadDispatcher _dispatcher = new MainThreadDispatcher();

        public string ModelPath
        {
            get => modelPath;
            set
            {
                if (IsLoaded || IsLoading)
                {
                    throw new InvalidOperationException("Cannot change model path after loading the model");
                }

                modelPath = value;
            }
        }
        public bool IsModelPathInStreamingAssets
        {
            get => isModelPathInStreamingAssets;
            set
            {
                if (IsLoaded || IsLoading)
                {
                    throw new InvalidOperationException("Cannot change model path after loading the model");
                }

                isModelPathInStreamingAssets = value;
            }
        }
        /// <summary>
        /// Checks if whisper weights are loaded and ready to be used.
        /// </summary>
        public bool IsLoaded => _whisper != null;
        /// <summary>
        /// Checks if whisper weights are still loading and not ready.
        /// </summary>
        public bool IsLoading { get; private set; }

        void Start()
        {
            if (!initOnAwake)
                return;

            InitModel();
        }
        void Update()
        {
            _dispatcher.Update();
        }

        /// <summary>
        /// Load model and default parameters. Prepare it for text transcription.
        /// </summary>
        public void InitModel()
        {
            // check if model is already loaded or actively loading
            if (IsLoaded)
            {
                Log.Warning(this, "Whisper model is already loaded and ready for use!");

                return;
            }

            if (IsLoading)
            {
                Log.Warning(this, "Whisper model is already loading!");

                return;
            }

            // load model and default params
            IsLoading = true;
            try
            {
                var path = isModelPathInStreamingAssets
                    ? Path.Combine(Application.streamingAssetsPath, modelPath)
                    : modelPath;

                _whisper = WhisperWrapper.InitFromFile(path, CreateContextParams());
                _params = WhisperParams.GetDefaultParams(strategy);

                UpdateParams();

                _whisper.OnNewSegment += OnNewSegmentHandler;
                _whisper.OnProgress += OnProgressHandler;
            }
            catch (Exception e)
            {
                Log.Error(this, e.Message);
            }

            IsLoading = false;
        }
        /// <summary>
        /// Checks if currently loaded whisper model supports multilingual transcription.
        /// </summary>
        public bool IsMultilingual()
        {
            if (!IsLoaded)
            {
                Log.Error(this, "Whisper model isn't loaded! Init Whisper model first!");

                return false;
            }

            return _whisper.IsMultilingual;
        }
        /// <summary>
        /// Start async transcription of audio buffer.
        /// </summary>
        /// <param name="samples">Raw audio buffer.</param>
        /// <param name="frequency">Audio sample rate.</param>
        /// <param name="channels">Audio channels count.</param>
        /// <returns>Full audio transcript. Null if transcription failed.</returns>
        public async void GetTextAsync(NativeArray<float> samples, int frequency, int channels)
        {
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
                return;

            UpdateParams();

            await _whisper.GetTextAsync(samples, frequency, channels, _params);
        }
        /// <summary>
        /// Create a new instance of Whisper streaming transcription.
        /// </summary>
        /// <param name="frequency">Audio sample rate.</param>
        /// <param name="channels">Audio channels count.</param>
        /// <returns>New streaming transcription. Null if failed.</returns>
        public async Task<WhisperStream> CreateStream(int frequency, int channels)
        {
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
            {
                Log.Error(this, "Model weights aren't loaded! Load model first!");

                return null;
            }

            var param = new WhisperStreamParams(_params, frequency, channels, stepSec, keepSec, lengthSec);

            return new WhisperStream(_whisper, param);
        }
        /// <summary>
        /// Create a new instance of Whisper streaming transcription from microphone input.
        /// </summary>
        /// <returns>New streaming transcription. Null if failed.</returns>
        public async Task<WhisperStream> CreateStream(MicrophoneRecord microphone)
        {
            var isLoaded = await CheckIfLoaded();
            if (!isLoaded)
            {
                Log.Error(this, "Model weights aren't loaded! Load model first!");

                return null;
            }

            // TODO: unity support only single input channel for microphone
            var param = new WhisperStreamParams(_params, microphone.frequency, 1, stepSec, keepSec, lengthSec);

            return new WhisperStream(_whisper, param, microphone);
        }

        void UpdateParams()
        {
            _params.Language = language;
            _params.Translate = translateToEnglish;
        }
        WhisperContextParams CreateContextParams()
        {
            var context = WhisperContextParams.GetDefaultParams();
            context.UseGpu = useGpu;
            context.FlashAttn = flashAttention;

            return context;
        }
        async Task<bool> CheckIfLoaded()
        {
            if (!IsLoaded && !IsLoading)
            {
                Log.Error(this, "Whisper model isn't loaded! Init Whisper model first!");

                return false;
            }

            // wait while model still loading
            while (IsLoading)
                await Task.Yield();

            return IsLoaded;
        }
        void OnNewSegmentHandler(WhisperSegment segment)
        {
            _dispatcher.Execute(() =>
            {
                OnNewSegment?.Invoke(segment);
            });
        }
        void OnProgressHandler(int progress)
        {
            _dispatcher.Execute(() =>
            {
                OnProgress?.Invoke(progress);
            });
        }
    }
}