using Core;

namespace Whisper
{
    public class WhisperContextParams
    {
        public WhisperNativeContextParams NativeParams => _param;
        private WhisperNativeContextParams _param;

        private WhisperContextParams(WhisperNativeContextParams param)
        {
            _param = param;
        }

        public bool UseGpu
        {
            get => _param.use_gpu;
            set => _param.use_gpu = value;
        }
        public bool FlashAttn
        {
            get => _param.flash_attn;
            set => _param.flash_attn = value;
        }
        public static WhisperContextParams GetDefaultParams()
        {
            var nativeParams = WhisperNative.whisper_context_default_params();
            Log.Info(nativeParams, "Default Whisper Context params generated!");

            return new WhisperContextParams(nativeParams);
        }
    }

    public class WhisperParams
    {
        public WhisperNativeParams NativeParams;

        public unsafe WhisperParams(WhisperNativeParams param)
        {
            NativeParams = param;
        }
    }
}