using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

using AOT;

using Core;

using Input;

using UI;

using Unity.Collections;

using UnityEngine;

namespace Whisper
{
    public class WhisperManager : MonoBehaviour
    {
        [MonoPInvokeCallback(typeof(whisper_new_segment_callback))]
        static void NewSegmentCallbackStatic(IntPtr ctx, IntPtr state, int nNew, IntPtr userDataPtr)
        {
            var userData = (WhisperUserData)GCHandle.FromIntPtr(userDataPtr).Target;
            userData.Wrapper.NewSegmentCallback(nNew);
        }

        [SerializeField] string ModelPath = "Whisper/";

        bool IsWorking = false;

        IntPtr Ctx = IntPtr.Zero;
        WhisperNativeContextParams ContextParams;
        WhisperNativeParams WhisperParams;

        [Space]
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
        [SerializeField] int MinFramesCount = 20000;
        [SerializeField] float BusyTime = 5f;

        bool isBusy;

        int Commands = 0;
        float BusyTimer;

        ConcurrentQueue<Task> Actions = new ConcurrentQueue<Task>();

        [Space]
        [SerializeField] RectTransform ModelSelectionPanel;
        [SerializeField] RectTransform ModelSelectionContent;
        [SerializeField] GameObject ModelButtonPrefab;

        List<MenuButton> ModelButtons = new List<MenuButton>();

        [Space]
        [SerializeField] VoiceCommander Commander;

        bool IsLoaded => Ctx != IntPtr.Zero;
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
            if (IsWorking)
                UnloadModel();
        }

        public void SwitchWorking()
        {
            if (!IsWorking)
                OpenModelFolderSelection();
            else
            {
                IsWorking = false;

                UnloadModel();
            }
        }
        public void InitModel(OuterInput input)
        {
            GetParams();

            Ctx = InitFromFile(input.Message);
            if (Ctx == IntPtr.Zero)
                Log.Error(this, $"Model {input.Agent} Initialization Error!");
            else
            {
                IsWorking = true;

                Log.Info(this, $"Model {input.Agent} initialized.");
            }

            CloseModelFolderSelection();
        }
        public async void GetText(NativeArray<float> samples, int frequency, int channels)
        {
            if (!IsWorking || IsBusy || samples.Length < MinFramesCount)
                return;

            IsBusy = true;
            Commands++;

            await InferenceWhisper(samples);
        }

        void OpenModelFolderSelection()
        {
            var folder = Path.Combine(Application.persistentDataPath, ModelPath);
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);

                Log.Warning(this, $"Model Folder was created, put Model Files inside\n{folder}\nor else!");

                return;
            }

            var models = Directory.GetFiles(folder, "*.bin");
            if (models == null || models.Length == 0)
            {
                Log.Warning(this, $"Path is empty!\n{folder}");

                return;
            }

            for (int m = 0; m < models.Length; m++)
            {
                var go = Instantiate(ModelButtonPrefab, ModelSelectionContent);
                var button = go.GetComponent<MenuButton>();
                var name = models[m].Replace(folder, "");
                button.SetLabel(name);
                button.AddInput(new OuterInput
                {
                    Title = "Model Picking",
                    Agent = name,
                    Message = models[m]
                });

                ModelButtons.Add(button);
            }

            ModelSelectionPanel.gameObject.SetActive(true);
        }
        void CloseModelFolderSelection()
        {
            ModelSelectionPanel.gameObject.SetActive(false);

            for (int m = 0; m < ModelButtons.Count; m++)
                Destroy(ModelButtons[m].gameObject);

            ModelButtons.Clear();
        }
        void UnloadModel()
        {
            if (IsLoaded)
            {
                WhisperNative.whisper_free(Ctx);

                Log.Info(this, $"Model unloaded.");
            }
            else
                Log.Error(this, $"Model Unloading Error!.");
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
        void ProcessVoiceText(WhisperSegment segment)
        {
            Log.Info(this, $"{segment.Text}");

            Commander.Process(segment.Text);
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
        void NewSegmentCallback(int nNew)
        {
            var nSegments = WhisperNative.whisper_full_n_segments(Ctx);
            var s0 = nSegments - nNew;

            for (var i = s0; i < nSegments; i++)
            {
                var segment = GetSegment(i);
                QueueActions(() => ProcessVoiceText(segment));
            }
        }
        IntPtr InitFromFile(string modelPath)
        {
            var buffer = FileUtils.ReadFile(modelPath);
            if (buffer == null)
                return IntPtr.Zero;

            return InitFromBuffer(buffer);
        }
        IntPtr InitFromBuffer(byte[] buffer)
        {
            var ctx = IntPtr.Zero;

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
        WhisperSegment GetSegment(int i)
        {
            var textPtr = WhisperNative.whisper_full_get_segment_text(Ctx, i);
            var text = TextUtils.StringFromNativeUtf8(textPtr);

            return new WhisperSegment(i, text);
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