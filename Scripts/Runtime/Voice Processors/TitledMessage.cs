using System;

using Core;

using Input;

using Unity.Entities;

using UnityEngine;

namespace Whisper
{
    [CreateAssetMenu(fileName = "Titled Message", menuName = "Whisper/Voice Processors/Titled Message")]
    public class TitledMessage : VoiceProcessor
    {
        public override string JSONPath => "Titled Messages/";
        public override VoiceCommand Command => command;

        [Space]
        public TitledMessageCommand command;
    }

    [Serializable]
    public class TitledMessageCommand : VoiceCommand
    {
        [Space]
        [LogInfo] public string Title;

        public override bool Invoke(string data)
        {
            var message = data.Trim().ToLower();
            if (message.StartsWith(Title, StringComparison.OrdinalIgnoreCase))
            {
                var input = new OuterInput(Input);
                input.Message = data.Replace(Title.ToLower(), "").Trim();

                Sys.Add_M(input, World.DefaultGameObjectInjectionWorld.EntityManager);

                return true;
            }

            return false;
        }
    }
}