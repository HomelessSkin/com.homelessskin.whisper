using System;
using System.Runtime.InteropServices;

using whisper_context_ptr = System.IntPtr;
using whisper_state_ptr = System.IntPtr;
using whisper_token = System.Int32;

// ReSharper disable InconsistentNaming
// ReSharper disable FieldCanBeMadeReadOnly.Local
// ReSharper disable IdentifierTypo
// ReSharper disable CommentTypo

using whisper_token_ptr = System.IntPtr;

namespace Whisper
{
    [UnmanagedFunctionPointer(CallingConvention.StdCall)]
    public delegate void whisper_new_segment_callback(whisper_context_ptr ctx, whisper_state_ptr state,
        int n_new, IntPtr user_data);

    public enum WhisperSamplingStrategy
    {
        WHISPER_SAMPLING_GREEDY = 0,
        WHISPER_SAMPLING_BEAM_SEARCH = 1,
    }

    public enum WhisperAlignmentHeadsPreset
    {
        WHISPER_AHEADS_NONE,
        WHISPER_AHEADS_N_TOP_MOST,
        WHISPER_AHEADS_CUSTOM,
        WHISPER_AHEADS_TINY_EN,
        WHISPER_AHEADS_TINY,
        WHISPER_AHEADS_BASE_EN,
        WHISPER_AHEADS_BASE,
        WHISPER_AHEADS_SMALL_EN,
        WHISPER_AHEADS_SMALL,
        WHISPER_AHEADS_MEDIUM_EN,
        WHISPER_AHEADS_MEDIUM,
        WHISPER_AHEADS_LARGE_V1,
        WHISPER_AHEADS_LARGE_V2,
        WHISPER_AHEADS_LARGE_V3,
        WHISPER_AHEADS_LARGE_V3_TURBO,
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct WhisperNativeTokenData
    {
        public whisper_token id;
        public whisper_token tid;

        public float p;
        public float plog;
        public float pt;
        public float ptsum;

        // token-level timestamp data (int64_t ‚ C)
        public long t0;
        public long t1;

        // [EXPERIMENTAL] Token-level timestamps with DTW (int64_t ‚ C)
        public long t_dtw;

        public float vlen;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct WhisperNativeAheads
    {
        UIntPtr n_heads;
        IntPtr heads;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct WhisperNativeContextParams
    {
        [MarshalAs(UnmanagedType.U1)] public bool use_gpu;
        [MarshalAs(UnmanagedType.U1)] public bool flash_attn;
        int gpu_device;

        // [EXPERIMENTAL] Token-level timestamps with DTW
        [MarshalAs(UnmanagedType.U1)] public bool dtw_token_timestamps;
        WhisperAlignmentHeadsPreset dtw_aheads_preset;

        int dtw_n_top;
        WhisperNativeAheads dtw_aheads;

        UIntPtr dtw_mem_size;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct greedy_struct
    {
        int best_of;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct beam_search_struct
    {
        int beam_size;
        float patience;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct WhisperVadParams
    {
        public float threshold;
        public int min_speech_duration_ms;
        public int min_silence_duration_ms;
        public float max_speech_duration_s;
        public int speech_pad_ms;
        public float samples_overlap;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct WhisperNativeParams
    {
        public WhisperSamplingStrategy strategy;

        public int n_threads;
        public int n_max_text_ctx;
        public int offset_ms;
        public int duration_ms;

        [MarshalAs(UnmanagedType.U1)] public bool translate;
        [MarshalAs(UnmanagedType.U1)] public bool no_context;
        [MarshalAs(UnmanagedType.U1)] public bool no_timestamps;
        [MarshalAs(UnmanagedType.U1)] public bool single_segment;
        [MarshalAs(UnmanagedType.U1)] public bool print_special;
        [MarshalAs(UnmanagedType.U1)] public bool print_progress;
        [MarshalAs(UnmanagedType.U1)] public bool print_realtime;
        [MarshalAs(UnmanagedType.U1)] public bool print_timestamps;

        // [EXPERIMENTAL] token-level timestamps
        [MarshalAs(UnmanagedType.U1)] public bool token_timestamps;
        float thold_pt;
        float thold_ptsum;
        int max_len;
        [MarshalAs(UnmanagedType.U1)] bool split_on_word;
        int max_tokens;

        // [EXPERIMENTAL] speed-up techniques
        [MarshalAs(UnmanagedType.U1)] bool debug_mode;
        public int audio_ctx;

        // [EXPERIMENTAL] [TDRZ] tinydiarize
        [MarshalAs(UnmanagedType.U1)] bool tdrz_enable;

        // A regular expression that matches tokens to suppress
        byte* suppress_regex;

        // tokens to provide to the whisper decoder as initial prompt
        public byte* initial_prompt;
        [MarshalAs(UnmanagedType.U1)] public bool carry_initial_prompt; // ÕŒ¬Œ≈ ‚ 1.9.x
        whisper_token_ptr prompt_tokens;
        int prompt_n_tokens;

        // for auto-detection, set to nullptr, "" or "auto"
        public byte* language;
        [MarshalAs(UnmanagedType.U1)] bool detect_language;

        // common decoding parameters:
        [MarshalAs(UnmanagedType.U1)] public bool suppress_blank;
        [MarshalAs(UnmanagedType.U1)] public bool suppress_nst; // œ≈–≈»Ã≈ÕŒ¬¿ÕŒ ËÁ suppress_non_speech_tokens

        float temperature;
        float max_initial_ts;
        float length_penalty;

        // fallback parameters
        public float temperature_inc;
        public float entropy_thold;
        public float logprob_thold;
        public float no_speech_thold;

        greedy_struct greedy;
        beam_search_struct beam_search;

        // called for every newly generated text segment
        public whisper_new_segment_callback new_segment_callback;
        public IntPtr new_segment_callback_user_data;

        // called on each progress update
        void* progress_callback;
        void* progress_callback_user_data;

        // called each time before the encoder starts
        void* encoder_begin_callback;
        void* encoder_begin_callback_user_data;

        // called each time before ggml computation starts
        void* abort_callback;
        void* abort_callback_user_data;

        // called by each decoder to filter obtained logits
        void* logits_filter_callback;
        void* logits_filter_callback_user_data;

        IntPtr grammar_rules;
        UIntPtr n_grammar_rules;
        UIntPtr i_start_rule;
        float grammar_penalty;

        // Voice Activity Detection (VAD) params ó ÕŒ¬€… ¡ÀŒ  ‚ 1.9.x
        [MarshalAs(UnmanagedType.U1)] public bool vad;
        public byte* vad_model_path;
        public WhisperVadParams vad_params;
    }
}