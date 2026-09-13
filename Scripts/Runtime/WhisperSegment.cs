using System;
using System.Collections.Generic;
using System.Text;

namespace Whisper
{
    public class WhisperSegment
    {
        public readonly int Index;
        public readonly string Text;

        public WhisperSegment(int index, string text)
        {
            Index = index;
            Text = text;
        }
    }
}