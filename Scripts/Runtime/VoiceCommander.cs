using System;
using System.Collections.Generic;
using System.IO;

using Core;

using UnityEngine;

namespace Whisper
{
    public class VoiceCommander : ResourceLoader
    {
        [Space]
        [SerializeField] string JSONFolder = "/Whisper";

        [Space]
        [SerializeField] VoiceProcessor[] Processors;

        public List<VoiceCommand> Commands = new List<VoiceCommand>();

        [Space]
        [SerializeField] char[] Trimming;

        void Start()
        {
            var path = Path.Combine(Application.persistentDataPath, JSONFolder);
            var folders = new List<string>();
            for (int p = 0; p < Processors.Length; p++)
            {
                var processor = Processors[p];
                Commands.Add(processor.Command);

                if (!folders.Contains(processor.JSONPath))
                {
                    folders.Add(processor.JSONPath);

                    var folder = Path.Combine(path, processor.JSONPath);
                    if (!Directory.Exists(folder))
                    {
                        Directory.CreateDirectory(folder);

                        continue;
                    }

                    var files = Directory.GetFiles(folder, "*.json");
                    if (files == null || files.Length == 0)
                        continue;

                    var type = processor.Command.GetType();
                    for (int f = 0; f < files.Length; f++)
                        AddCommand(files[f], type);
                }
            }
        }

        public void Process(string data)
        {
            data = data.Trim(Trimming).ToLower();
            for (int c = 0; c < Commands.Count; c++)
                if (Commands[c].Invoke(data))
                    return;
        }

        void AddCommand(string file, Type type)
        {
            var text = File.ReadAllText(file);
            if (!string.IsNullOrEmpty(text))
            {
                var obj = JsonUtility.FromJson(text, type);
                Commands.Add(obj as VoiceCommand);

                Log.Info(this, $"Added Command of Type: {type.FullName}");
                Log.Object(this, obj as ILogTarget);
            }
        }

#if UNITY_EDITOR
        protected override void LoadResources() => Load<VoiceProcessor>(ref Processors);
#endif
    }
}