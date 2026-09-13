using System;
using System.Collections.Concurrent;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using AOT;

using Core;

using Unity.Collections;

using UnityEngine;

namespace Whisper
{
    public class WhisperManager : MonoBehaviour
    {
        [Header("Model")]
        [SerializeField] string ModelPath = "Whisper/ggml-tiny.bin";

        [Header("Inference")]
        [SerializeField] WhisperSamplingStrategy Strategy = WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY;
        [SerializeField] bool TranslateToEnglish;
        [SerializeField] string Language = "en";
        [SerializeField] bool UseGpu;
        [SerializeField] bool FlashAttention;
        [SerializeField] int ThreadsNumber = 1;
        [SerializeField] float TemperatureInc;
        [SerializeField] float EntropyThold;
        [SerializeField] float LogprobThold;
        [SerializeField] float NoSpeechThold;

        [Space]
        [SerializeField] float BusyTime = 5f;

        bool isBusy;
        bool IsBusy
        {
            get => isBusy;
            set
            {
                if (value || Commands > 0)
                {
                    isBusy = true;
                    BusyTimer = BusyTime;

                    return;
                }

                isBusy = value;
            }
        }

        int Commands = 0;
        float BusyTimer;

        WhisperNativeContextParams ContextParams;
        WhisperNativeParams WhisperParams;

        ConcurrentQueue<Task> Actions = new ConcurrentQueue<Task>();

        bool IsLoaded => Ctx != IntPtr.Zero;
        IntPtr Ctx = IntPtr.Zero;

        void Start()
        {
            InitModel();
        }
        void Update()
        {
            QueueUpdate();

            if (BusyTimer > 0f)
                BusyTimer -= Time.deltaTime;
            else
                IsBusy = false;
        }
        void OnDestroy()
        {
            if (IsLoaded)
                WhisperNative.whisper_free(Ctx);
        }

        public async void GetText(NativeArray<float> samples, int frequency, int channels)
        {
            if (!CheckLoaded() || IsBusy || samples.Length < 20000)
                return;

            IsBusy = true;
            Commands++;

            await InferenceWhisper(samples);
        }

        async Task InferenceWhisper(NativeArray<float> samples)
        {
            var array = samples.ToArray();

            Log.Info(this, $"Starting Inference with {samples.Length} samples.");

            await Task.Run(() =>
            {
                unsafe
                {
                    fixed (float* samplesPtr = array)
                    {
                        var code = WhisperNative.whisper_full(Ctx, WhisperParams, samplesPtr, samples.Length);
                        if (code != 0)
                            Log.Error(this, $"Whisper failed to process data! Error code: {code}.");
                    }
                }
            });
        }

        void InitModel()
        {
            if (IsLoaded)
            {
                Log.Warning(this, "Whisper model is already loaded and ready for use!");

                return;
            }

            GetParams();

            Ctx = InitFromFile(Path.Combine(Application.persistentDataPath, ModelPath));
            if (Ctx == IntPtr.Zero)
                Log.Error(this, $"Model Initialization Error!");
        }
        void GetParams()
        {
            ContextParams = WhisperNative.whisper_context_default_params();

            ContextParams.use_gpu = UseGpu;
            ContextParams.flash_attn = FlashAttention;

            WhisperParams = WhisperNative.whisper_full_default_params(Strategy);

            unsafe
            {
                WhisperParams.language = (byte*)Marshal.StringToHGlobalAnsi(Language);
            }

            WhisperParams.translate = TranslateToEnglish;

            WhisperParams.n_threads = ThreadsNumber;
            WhisperParams.temperature_inc = TemperatureInc;
            WhisperParams.entropy_thold = EntropyThold;
            WhisperParams.logprob_thold = LogprobThold;
            WhisperParams.no_speech_thold = NoSpeechThold;

            WhisperParams.no_context =
            WhisperParams.no_timestamps =
            WhisperParams.single_segment =
            true;

            WhisperParams.print_special =
            WhisperParams.print_progress =
            WhisperParams.print_realtime =
            WhisperParams.print_timestamps =
            false;

            var userData = new WhisperUserData(this);

            WhisperParams.new_segment_callback = NewSegmentCallbackStatic;
            WhisperParams.new_segment_callback_user_data = GCHandle.ToIntPtr(GCHandle.Alloc(userData));

            Log.Info(this, "Default params generated!");
        }
        void LogText(WhisperSegment segment)
        {
            Log.Info(this, $"{segment.Text}");
        }
        void QueueUpdate()
        {
            while (Actions.TryDequeue(out var task))
            {
                Commands--;

                task.RunSynchronously();
            }
        }
        void QueueActions(Action action)
        {
            Actions.Enqueue(new Task(action));
        }
        bool CheckLoaded()
        {
            if (!IsLoaded)
            {
                Log.Error(this, "Whisper model isn't loaded! Init Whisper model first!");

                return false;
            }

            return true;
        }
        WhisperSegment GetSegment(int i)
        {
            var textPtr = WhisperNative.whisper_full_get_segment_text(Ctx, i);
            var text = TextUtils.StringFromNativeUtf8(textPtr);

            return new WhisperSegment(i, text);
        }

        [MonoPInvokeCallback(typeof(whisper_new_segment_callback))]
        static void NewSegmentCallbackStatic(IntPtr ctx, IntPtr state, int nNew, IntPtr userDataPtr)
        {
            var userData = (WhisperUserData)GCHandle.FromIntPtr(userDataPtr).Target;
            userData.Wrapper.NewSegmentCallback(nNew);
        }
        void NewSegmentCallback(int nNew)
        {
            var nSegments = WhisperNative.whisper_full_n_segments(Ctx);
            var s0 = nSegments - nNew;

            for (var i = s0; i < nSegments; i++)
            {
                var segment = GetSegment(i);
                QueueActions(() => LogText(segment));
            }
        }

        IntPtr InitFromFile(string modelPath)
        {
            Log.Info(this, $"Trying to load Whisper model from {modelPath}...");

            var buffer = FileUtils.ReadFile(modelPath);
            if (buffer == null)
                return IntPtr.Zero;

            return InitFromBuffer(buffer);
        }
        IntPtr InitFromBuffer(byte[] buffer)
        {
            var ctx = IntPtr.Zero;

            Log.Info(this, $"Trying to load Whisper model from buffer...");

            if (buffer == null || buffer.Length == 0)
            {
                Log.Error(this, "Whisper model buffer is null or empty!");

                return ctx;
            }

            var length = new UIntPtr((uint)buffer.Length);

            unsafe
            {
                fixed (byte* bufferPtr = buffer)
                {
                    ctx = WhisperNative.whisper_init_from_buffer_with_params((IntPtr)bufferPtr, length, ContextParams);
                }
            }

            return ctx;
        }
        async Task<IntPtr> InitFromFileAsync(string modelPath)
        {
            Log.Info(this, $"Trying to load Whisper model from {modelPath}...");

            var buffer = await FileUtils.ReadFileAsync(modelPath);
            if (buffer == null)
                return IntPtr.Zero;

            return await InitFromBufferAsync(buffer);
        }
        async Task<IntPtr> InitFromBufferAsync(byte[] buffer)
        {
            return await Task.Factory.StartNew(() => InitFromBuffer(buffer));
        }

        struct WhisperUserData
        {
            public readonly WhisperManager Wrapper;

            public WhisperUserData(WhisperManager wrapper)
            {
                Wrapper = wrapper;
            }
        }
    }
}