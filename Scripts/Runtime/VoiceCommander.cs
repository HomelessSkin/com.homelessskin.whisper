using Core;

using UnityEngine;

namespace Whisper
{
    public class VoiceCommander : ResourceLoader
    {
        [Space]
        [SerializeField] VoiceProcessor[] Processors;

        [Space]
        [SerializeField] char[] Trimming;

        public void Process(string data)
        {
            data = data.Trim(Trimming).ToLower();
            for (int p = 0; p < Processors.Length; p++)
                if (Processors[p].Process(data))
                    return;
        }

#if UNITY_EDITOR
        protected override void LoadResources() => Load<VoiceProcessor>(ref Processors);
#endif
    }
}