using System;
using System.Linq;

using Core;

using TMPro;

using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using Unity.Jobs.LowLevel.Unsafe;

using UnityEngine;

namespace Whisper
{
    public class VoiceCommandRecorder : MonoBehaviour
    {
        State state = State.Idle;

        [SerializeField] int SampleRate = 16000;
        [SerializeField] int BufferLengthSec = 10;
        [SerializeField] int AnalysisWindow = 1024;
        [SerializeField] float PreBufferSec = 0.2f;
        [SerializeField] float MaxCommandDuration = 5f;

        float RecordingStartTime;

        AudioClip MicClip;

        [Space]
        [SerializeField] float StartThresholdDb = -30f;
        [SerializeField] float StartHoldTime = 0.1f;

        int StartSample;
        float StartHoldTimer;

        [Space]
        [SerializeField] float SilenceThresholdDb = -40f;
        [SerializeField] float SilenceDuration = 1.0f;

        int LastVoiceSample;
        float SilenceTimer;

        int TotalSamples;

        [Space]
        [SerializeField] Vector3 MicLevelOrigin;
        [SerializeField] Vector3 MicLevelVelocity;
        [SerializeField] RectTransform MicLevel;
        [SerializeField] TMP_Text MicLevelText;

        [Space]
        [SerializeField] bool Echo;
        [SerializeField] string MicrophoneDefaultLabel = "Default microphone";
        [SerializeField] TMP_Dropdown MicrophoneDropdown;

        int SelectedMicDevice;
        string MicName => MicrophoneDropdown.options[SelectedMicDevice].text;

        [Space]
        [SerializeField] WhisperManager Manager;

        void Start()
        {
            RefreshMicrophones();
        }
        void Update()
        {
            if (!MicClip)
                return;

            var currentPos = Microphone.GetPosition(MicName);
            var dB = GetCurrentDb(currentPos);

            MicLevel.anchoredPosition = MicLevelOrigin + dB * MicLevelVelocity;
            MicLevelText.text = dB.ToString("0.0 dB");

            switch (state)
            {
                case State.Idle:
                if (dB > StartThresholdDb)
                {
                    StartHoldTimer += Time.deltaTime;
                    if (StartHoldTimer >= StartHoldTime)
                    {
                        state = State.Recording;

                        var preSamples = (int)(PreBufferSec * SampleRate);
                        StartSample = (currentPos - preSamples + TotalSamples) % TotalSamples;
                        LastVoiceSample = currentPos;
                        SilenceTimer = 0f;
                        RecordingStartTime = Time.time;
                    }
                }
                else
                    StartHoldTimer = 0f;
                break;

                case State.Recording:
                if (dB > SilenceThresholdDb)
                {
                    LastVoiceSample = currentPos;
                    SilenceTimer = 0f;
                }
                else if (Time.time - RecordingStartTime >= MaxCommandDuration)
                    EndRecording(currentPos);
                else
                {
                    SilenceTimer += Time.deltaTime;
                    if (SilenceTimer >= SilenceDuration)
                        EndRecording(LastVoiceSample);
                }
                break;
            }
        }
        void OnDestroy()
        {
            if (Microphone.IsRecording(MicName))
                Microphone.End(MicName);
        }

        public void RefreshMicrophones()
        {
            MicrophoneDropdown.options = Microphone
                .devices
                .Prepend(MicrophoneDefaultLabel)
                .Select(text => new TMP_Dropdown.OptionData(text))
                .ToList();

            if (SelectedMicDevice < MicrophoneDropdown.options.Count)
                MicrophoneDropdown.value = SelectedMicDevice;
            else
                MicrophoneDropdown.value = SelectedMicDevice = 0;
        }
        public void OnMicrophoneChanged(int ind)
        {
            if (Microphone.IsRecording(MicName))
                Microphone.End(MicName);

            SelectedMicDevice = ind;
            if (SelectedMicDevice > 0)
            {
                MicClip = Microphone.Start(MicName, true, BufferLengthSec, SampleRate);
                TotalSamples = MicClip.samples * MicClip.channels;
            }
            else
            {
                MicLevel.anchoredPosition = MicLevelOrigin;
                MicLevelText.text = "0.0 dB";
            }
        }

        void EndRecording(int endSample)
        {
            state = State.Idle;
            StartHoldTimer = SilenceTimer = 0f;

            var length = (endSample - StartSample + TotalSamples) % TotalSamples;
            if (length <= 0)
                return;

            var fullBuffer = new NativeArray<float>(TotalSamples, Allocator.TempJob);
            if (!MicClip.GetData(fullBuffer, 0))
            {
                Log.Error(this, "Getting Data from Mic Clip Error!");

                fullBuffer.Dispose();

                return;
            }

            var result = new NativeArray<float>(length, Allocator.TempJob);

            new BufferingJob
            {
                Start = StartSample,
                Total = TotalSamples,
                Source = fullBuffer,

                Result = result,
            }
            .Schedule(length, length / JobsUtility.JobWorkerCount)
            .Complete();

            Manager.GetText(result, MicClip.frequency, MicClip.channels);

            if (Echo)
            {
                var clip = AudioClip.Create("echo", result.Length, MicClip.channels, MicClip.frequency, false);
                if (clip.SetData(result, 0))
                    PlayAudioAndDestroy.Play(clip, Vector3.zero);
                else
                    Destroy(clip);
            }

            result.Dispose();
            fullBuffer.Dispose();
        }
        float GetCurrentDb(int currentPos)
        {
            var start = currentPos - AnalysisWindow;
            if (start < 0)
                start += TotalSamples;

            var samples = new float[AnalysisWindow];

            if (start + AnalysisWindow <= TotalSamples)
                MicClip.GetData(samples, start);
            else
            {
                var firstPart = TotalSamples - start;
                var part1 = new float[firstPart];

                MicClip.GetData(part1, start);

                var part2 = new float[AnalysisWindow - firstPart];
                MicClip.GetData(part2, 0);

                Array.Copy(part1, 0, samples, 0, firstPart);
                Array.Copy(part2, 0, samples, firstPart, part2.Length);
            }

            var sum = 0f;
            for (int i = 0; i < samples.Length; i++)
                sum += samples[i] * samples[i];

            var rms = Mathf.Sqrt(sum / samples.Length);
            if (rms < 1e-7f)
                return -160f;

            return 20f * Mathf.Log10(rms);
        }

        enum State : byte
        {
            Idle = 0,
            Recording = 1
        }

        [BurstCompile]
        struct BufferingJob : IJobParallelFor
        {
            [ReadOnly] public int Start;
            [ReadOnly] public int Total;

            [ReadOnly] public NativeArray<float> Source;

            [WriteOnly] public NativeArray<float> Result;

            public void Execute(int index) => Result[index] = Source[(Start + index) % Total];
        }
    }
}