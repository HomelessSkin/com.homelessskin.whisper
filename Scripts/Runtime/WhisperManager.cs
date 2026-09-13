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
        [SerializeField] bool PathInStreamingAssets = true;
        [SerializeField] string ModelPath = "Whisper/ggml-tiny.bin";

        [Header("Inference")]
        [SerializeField] bool UseGpu;
        [SerializeField] bool FlashAttention;

        [Header("Language")]
        [SerializeField] bool TranslateToEnglish;
        [SerializeField] string Language = "en";

        [Header("Advanced settings")]
        [SerializeField] WhisperSamplingStrategy Strategy = WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY;

        [Space]
        [SerializeField] float BusyTime = 5f;

        float BusyTimer;
        IntPtr Ctx = IntPtr.Zero;

        WhisperParams Params;

        ConcurrentQueue<Task> Actions = new ConcurrentQueue<Task>();

        bool IsLoaded => Ctx != IntPtr.Zero;

        void Start()
        {
            InitModel();
        }
        void Update()
        {
            QueueUpdate();

            if (BusyTimer > 0f)
                BusyTimer -= Time.deltaTime;
        }

        public async void GetText(NativeArray<float> samples, int frequency, int channels)
        {
            if (!CheckLoaded() || BusyTimer > 0f || Actions.Count > 0 || samples.Length < 20000)
                return;

            BusyTimer += BusyTime;

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
                        var code = WhisperNative.whisper_full(Ctx, Params.NativeParams, samplesPtr, samples.Length);
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

            var path = PathInStreamingAssets
                ? Path.Combine(Application.streamingAssetsPath, ModelPath)
                : ModelPath;

            Ctx = InitFromFile(path, CreateContextParams());
            if (Ctx == IntPtr.Zero)
                Log.Error(this, $"Error!");
            else
                GetParams();
        }
        void GetParams()
        {
            var nativeParams = WhisperNative.whisper_full_default_params(Strategy);

            var userData = new WhisperUserData(this);

            if (nativeParams.new_segment_callback == null &&
                 nativeParams.new_segment_callback_user_data == IntPtr.Zero)
            {
                nativeParams.new_segment_callback = NewSegmentCallbackStatic;
                nativeParams.new_segment_callback_user_data = GCHandle.ToIntPtr(GCHandle.Alloc(userData));
            }

            nativeParams.translate = TranslateToEnglish;

            nativeParams.n_threads = 1;
            nativeParams.n_max_text_ctx = 0;
            nativeParams.no_context =
            nativeParams.single_segment =
            true;

            nativeParams.print_progress =
            nativeParams.print_realtime =
            nativeParams.print_timestamps =
            false;

            unsafe
            {
                nativeParams.language = (byte*)Marshal.StringToHGlobalAnsi(Language);
            }

            Params = new WhisperParams(nativeParams);

            Log.Info(nativeParams, "Default params generated!");
        }
        void LogText(WhisperSegment segment)
        {
            Log.Info(this, $"{segment.Text}");
        }
        void QueueUpdate()
        {
            while (Actions.TryDequeue(out var task))
            {
                BusyTimer = BusyTime;

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
        WhisperContextParams CreateContextParams()
        {
            var context = WhisperContextParams.GetDefaultParams();
            context.UseGpu = UseGpu;
            context.FlashAttn = FlashAttention;

            return context;
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

        IntPtr InitFromFile(string modelPath, WhisperContextParams contextParams)
        {
            // load model weights
            Log.Info(contextParams, $"Trying to load Whisper model from {modelPath}...");
            var buffer = FileUtils.ReadFile(modelPath);
            if (buffer == null)
                return IntPtr.Zero;

            return InitFromBuffer(buffer, contextParams);
        }
        IntPtr InitFromBuffer(byte[] buffer, WhisperContextParams contextParams)
        {
            var ctx = IntPtr.Zero;
            Log.Info(contextParams, $"Trying to load Whisper model from buffer...");
            if (buffer == null || buffer.Length == 0)
            {
                Log.Error(contextParams, "Whisper model buffer is null or empty!");

                return ctx;
            }

            // we need to write buffer length as size_t
            // UIntPtr will work because size_t is size of pointer
            var length = new UIntPtr((uint)buffer.Length);

            unsafe
            {
                // this only works because whisper makes copy of the buffer
                fixed (byte* bufferPtr = buffer)
                {
                    ctx = WhisperNative.whisper_init_from_buffer_with_params((IntPtr)bufferPtr,
                        length, contextParams.NativeParams);
                }
            }

            return ctx;
        }
        async Task<IntPtr> InitFromFileAsync(string modelPath, WhisperContextParams contextParams)
        {
            Log.Info(contextParams, $"Trying to load Whisper model from {modelPath}...");
            var buffer = await FileUtils.ReadFileAsync(modelPath);
            if (buffer == null)
                return IntPtr.Zero;

            return await InitFromBufferAsync(buffer, contextParams);
        }
        async Task<IntPtr> InitFromBufferAsync(byte[] buffer, WhisperContextParams contextParams)
        {
            return await Task.Factory.StartNew(() => InitFromBuffer(buffer, contextParams));
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