using Core;

using Input;

using UnityEngine;

namespace Whisper
{
    public abstract class VoiceProcessor : KeyScriptable
    {
        [Space]
        public OuterInput Input;

        public abstract bool Process(string data);

#if UNITY_EDITOR
        protected override void Reset()
        {
            base.Reset();

            Input = new OuterInput();
            Input.Source = "Voice";
            Input.Agent = "Speaker";
        }
#endif
    }
}