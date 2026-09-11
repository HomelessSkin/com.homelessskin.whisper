using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using AOT;

using Core;

using Unity.Collections;

using UnityEngine;

using Whisper.Native;
using Whisper.Utils;

namespace Whisper
{
    public delegate void OnNewSegmentDelegate(WhisperSegment text);
    public delegate void OnProgressDelegate(int progress);

    /// <summary>
    /// Wrapper for loaded whisper model.
    /// </summary>
    public class WhisperWrapper
    {
        public const int WhisperSampleRate = 16000;

        /// <summary>
        /// Raised when whisper transcribed a new text segment from audio. 
        /// </summary>
        /// <remarks>Use <see cref="MainThreadDispatcher"/> for handling event in Unity main thread.</remarks>
        public event OnNewSegmentDelegate OnNewSegment;
        /// <summary>
        /// Raised when whisper made some progress in transcribing audio.
        /// Progress changes from 0 to 100 included.
        /// </summary>
        /// <remarks>Use <see cref="MainThreadDispatcher"/> for handling event in Unity main thread.</remarks>
        public event OnProgressDelegate OnProgress;

        readonly IntPtr _whisperCtx;
        readonly WhisperNativeParams _params;
        readonly object _lock = new object();

        WhisperWrapper(IntPtr whisperCtx)
        {
            _whisperCtx = whisperCtx;
        }

        ~WhisperWrapper()
        {
            if (_whisperCtx == IntPtr.Zero)
                return;

            WhisperNative.whisper_free(_whisperCtx);
        }

        /// <summary>
        /// Checks if currently loaded whisper model supports multilingual transcription.
        /// </summary>
        public bool IsMultilingual => WhisperNative.whisper_is_multilingual(_whisperCtx) != 0;

        /// <summary>
        /// Transcribe audio buffer. Will block thread until transcription complete.
        /// </summary>
        /// <param name="samples">Raw audio buffer.</param>
        /// <param name="frequency">Audio sample rate.</param>
        /// <param name="channels">Audio channels count.</param>
        /// <param name="param">Whisper inference parameters.</param>
        /// <returns>Full audio transcript. Null if transcription failed.</returns>
        public async Task GetTextAsync(NativeArray<float> samples, int frequency, int channels, WhisperParams param)
        {
            {
                // preprocess data if necessary
                Log.Info(this, "Preprocessing audio data...");

                //var readySamples = AudioUtils.Preprocess(samples, frequency, channels, WhisperSampleRate);

                var userData = new WhisperUserData(this, param);
                var gch = GCHandle.Alloc(userData);
                var nativeParams = param.NativeParams;

                // add callback (if no custom callback set)
                if (nativeParams.new_segment_callback == null &&
                    nativeParams.new_segment_callback_user_data == IntPtr.Zero)
                {
                    nativeParams.new_segment_callback = NewSegmentCallbackStatic;
                    nativeParams.new_segment_callback_user_data = GCHandle.ToIntPtr(gch);
                }

                if (nativeParams.progress_callback == null &&
                    nativeParams.progress_callback_user_data == IntPtr.Zero)
                {
                    nativeParams.progress_callback = ProgressCallbackStatic;
                    nativeParams.progress_callback_user_data = GCHandle.ToIntPtr(gch);
                }

                // start inference
                await InferenceWhisper(samples, nativeParams);

                gch.Free();
            }
        }

        async Task InferenceWhisper(NativeArray<float> samples, WhisperNativeParams param)
        {
            var array = samples.ToArray();

            await Task.Run(() =>
            {
                unsafe
                {
                    fixed (float* samplesPtr = array)
                    {
                        var code = WhisperNative.whisper_full(_whisperCtx, param, samplesPtr, samples.Length);
                        if (code != 0)
                        {
                            Log.Error(this, $"Whisper failed to process data! Error code: {code}.");

                            return false;
                        }
                    }
                }

                return true;
            });
        }
        WhisperSegment GetSegment(int i, WhisperParams param)
        {
            // get segment text and timestamps
            var textPtr = WhisperNative.whisper_full_get_segment_text(_whisperCtx, i);
            var text = TextUtils.StringFromNativeUtf8(textPtr);
            var start = WhisperNative.whisper_full_get_segment_t0(_whisperCtx, i);
            var end = WhisperNative.whisper_full_get_segment_t1(_whisperCtx, i);
            var segment = new WhisperSegment(i, text, start, end);

            // get all tokens
            var tokensN = WhisperNative.whisper_full_n_tokens(_whisperCtx, i);
            segment.Tokens = new WhisperTokenData[tokensN];
            for (var j = 0; j < tokensN; j++)
            {
                var nativeToken = WhisperNative.whisper_full_get_token_data(_whisperCtx, i, j);
                var textTokenPtr = WhisperNative.whisper_full_get_token_text(_whisperCtx, i, j);
                var textToken = TextUtils.StringFromNativeUtf8(textTokenPtr);
                var isSpecial = nativeToken.id >= WhisperNative.whisper_token_eot(_whisperCtx);
                var token = new WhisperTokenData(nativeToken, textToken, isSpecial);
                segment.Tokens[j] = token;
            }

            return segment;
        }

        [MonoPInvokeCallback(typeof(whisper_new_segment_callback))]
        static void NewSegmentCallbackStatic(IntPtr ctx, IntPtr state, int nNew, IntPtr userDataPtr)
        {
            // relay this static function to wrapper instance
            var userData = (WhisperUserData)GCHandle.FromIntPtr(userDataPtr).Target;
            userData.Wrapper.NewSegmentCallback(nNew, userData.Param);
        }
        void NewSegmentCallback(int nNew, WhisperParams param)
        {
            // start reading new segments
            var nSegments = WhisperNative.whisper_full_n_segments(_whisperCtx);
            var s0 = nSegments - nNew;
            for (var i = s0; i < nSegments; i++)
            {
                var segment = GetSegment(i, param);

                OnNewSegment?.Invoke(segment);
            }
        }

        [MonoPInvokeCallback(typeof(whisper_progress_callback))]
        static void ProgressCallbackStatic(IntPtr ctx, IntPtr state, int progress, IntPtr userDataPtr)
        {
            // relay this static function to wrapper instance
            var userData = (WhisperUserData)GCHandle.FromIntPtr(userDataPtr).Target;
            userData.Wrapper.ProgressCallback(progress);
        }
        void ProgressCallback(int progress)
        {
            OnProgress?.Invoke(progress);
        }

        /// <summary>
        /// Loads whisper model from file path with default context params.
        /// </summary>
        /// <param name="modelPath">Absolute file path to model weights.</param>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static WhisperWrapper InitFromFile(string modelPath)
        {
            var param = WhisperContextParams.GetDefaultParams();
            return InitFromFile(modelPath, param);
        }
        /// <summary>
        /// Loads whisper model from file path.
        /// </summary>
        /// <param name="modelPath">Absolute file path to model weights.</param>
        /// <param name="contextParams">Whisper context params used during model loading.</param>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static WhisperWrapper InitFromFile(string modelPath, WhisperContextParams contextParams)
        {
            // load model weights
            Log.Info(contextParams, $"Trying to load Whisper model from {modelPath}...");
            var buffer = FileUtils.ReadFile(modelPath);
            if (buffer == null)
                return null;

            return InitFromBuffer(buffer, contextParams);
        }
        /// <summary>
        /// Start async loading of whisper model from file path with default context params.
        /// </summary>
        /// <param name="modelPath">Absolute file path to model weights.</param>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static async Task<WhisperWrapper> InitFromFileAsync(string modelPath)
        {
            var param = WhisperContextParams.GetDefaultParams();
            return await InitFromFileAsync(modelPath, param);
        }
        /// <summary>
        /// Start async loading of whisper model from file path.
        /// </summary>
        /// <param name="modelPath">Absolute file path to model weights.</param>
        /// <param name="contextParams">Whisper context params used during model loading.</param>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static async Task<WhisperWrapper> InitFromFileAsync(string modelPath, WhisperContextParams contextParams)
        {
            Log.Info(contextParams, $"Trying to load Whisper model from {modelPath}...");
            var buffer = await FileUtils.ReadFileAsync(modelPath);
            if (buffer == null)
                return null;

            return await InitFromBufferAsync(buffer, contextParams);
        }
        /// <summary>
        /// Loads whisper model from byte buffer with default context params.
        /// </summary>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static WhisperWrapper InitFromBuffer(byte[] buffer)
        {
            var param = WhisperContextParams.GetDefaultParams();
            return InitFromBuffer(buffer, param);
        }
        /// <summary>
        /// Loads whisper model from byte buffer.
        /// </summary>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static WhisperWrapper InitFromBuffer(byte[] buffer, WhisperContextParams contextParams)
        {
            Log.Info(contextParams, $"Trying to load Whisper model from buffer...");
            if (buffer == null || buffer.Length == 0)
            {
                Log.Error(contextParams, "Whisper model buffer is null or empty!");

                return null;
            }

            // we need to write buffer length as size_t
            // UIntPtr will work because size_t is size of pointer
            var length = new UIntPtr((uint)buffer.Length);

            IntPtr ctx;
            unsafe
            {
                // this only works because whisper makes copy of the buffer
                fixed (byte* bufferPtr = buffer)
                {
                    ctx = WhisperNative.whisper_init_from_buffer_with_params((IntPtr)bufferPtr,
                        length, contextParams.NativeParams);
                }
            }

            if (ctx == IntPtr.Zero)
            {
                Log.Error(ctx, "Failed to load Whisper model!");

                return null;
            }

            return new WhisperWrapper(ctx);
        }
        /// <summary>
        /// Start async loading of whisper model from byte buffer with default context params.
        /// </summary>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static async Task<WhisperWrapper> InitFromBufferAsync(byte[] buffer)
        {
            var param = WhisperContextParams.GetDefaultParams();
            return await InitFromBufferAsync(buffer, param);
        }
        /// <summary>
        /// Start async loading of whisper model from byte buffer.
        /// </summary>
        /// <returns>Loaded whisper model. Null if loading failed.</returns>
        public static async Task<WhisperWrapper> InitFromBufferAsync(byte[] buffer, WhisperContextParams contextParams)
        {
            var asyncTask = Task.Factory.StartNew(() => InitFromBuffer(buffer, contextParams));
            return await asyncTask;
        }
        /// <summary>
        /// Get human readable information about what extensions compiled library expects.
        /// It will check if Whisper expects for AVX, CUDA, CoreML, etc.
        /// </summary>
        /// <remarks>
        /// It doesnt mean your hardware support it. It means that library expects
        /// your hardware to support it. For example, CPU which doesn't support
        /// AVX will still print "AVX=1", because library was compiled to expect AVX.
        /// </remarks>
        public static string GetSystemInfo()
        {
            var systemInfoPtr = WhisperNative.whisper_print_system_info();
            Log.Info(systemInfoPtr, "System information recived!");

            return TextUtils.StringFromNativeUtf8(systemInfoPtr);
        }

        struct WhisperUserData
        {
            public readonly WhisperWrapper Wrapper;
            public readonly WhisperParams Param;

            public WhisperUserData(WhisperWrapper wrapper, WhisperParams param)
            {
                Wrapper = wrapper;
                Param = param;
            }
        }
    }
}