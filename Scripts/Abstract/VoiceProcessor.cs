using System;

using Core;

using Input;

using UnityEngine;

namespace Whisper
{
    public abstract class VoiceProcessor : KeyScriptable, ILogTarget
    {
        public abstract string JSONPath { get; }
        [LogInfo] public abstract VoiceCommand Command { get; }

#if UNITY_EDITOR
        [Space]
        [TextArea(20, 50)] public string JSON;

        protected override void Reset()
        {
            base.Reset();

            Command.Input.Source = "Voice";
            Command.Input.Agent = "Speaker";
        }

        void OnValidate()
        {
            JSON = JsonUtility.ToJson(Command, true);
        }
#endif
    }

    [Serializable]
    public abstract class VoiceCommand : ILogTarget
    {
        [LogInfo] public OuterInput Input;

        public abstract bool Invoke(string data);
    }
}