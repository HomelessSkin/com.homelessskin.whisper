using System;

using Core;

using Unity.Entities;

using UnityEngine;

namespace Whisper
{
    [CreateAssetMenu(fileName = "Container Checker", menuName = "Whisper/Voice Processors/Container Checker")]
    public class ContainerChecker : VoiceProcessor
    {
        public override string JSONPath => "Container Checkers/";
        public override VoiceCommand Command => command;

        [Space]
        public ContainerCheckerCommand command;
    }

    [Serializable]
    public class ContainerCheckerCommand : VoiceCommand
    {
        [Space]
        [LogInfo] public string[] Keys;

        public override bool Invoke(string data)
        {
            var contains = true;
            for (int k = 0; k < Keys.Length; k++)
            {
                contains &= data.Contains(Keys[k].ToLower());

                if (!contains)
                    break;
            }

            if (contains)
                Sys.Add_M(Input, World.DefaultGameObjectInjectionWorld.EntityManager);

            return contains;
        }
    }
}