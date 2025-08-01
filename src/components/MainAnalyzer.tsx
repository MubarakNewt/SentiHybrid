import React, { useState } from 'react';
import { Send, Loader2, MessageSquare } from 'lucide-react';
import { analyzeSentiment } from '../utils/sentimentAnalysis';

interface MainAnalyzerProps {
  onAnalysisComplete: (analysis: any) => void;
}

const MainAnalyzer: React.FC<MainAnalyzerProps> = ({ onAnalysisComplete }) => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim() || isAnalyzing) return;

    setIsAnalyzing(true);
    
    setTimeout(() => {
      const analysis = analyzeSentiment(text);
      onAnalysisComplete({
        ...analysis,
        originalText: text,
        timestamp: new Date().toISOString()
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleAnalyze();
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
      <div className="flex items-center space-x-3 mb-6">
        <MessageSquare className="h-6 w-6 text-blue-600" />
        <h2 className="text-xl font-semibold text-gray-900">Text Analysis</h2>
      </div>

      <div className="space-y-6">
        <div>
          <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-3">
            Enter text for sentiment analysis
          </label>
          <textarea
            id="text-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your text here... (e.g., tweets, reviews, comments)"
            rows={6}
            className="w-full border border-gray-300 rounded-xl px-4 py-3 text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
          />
          <div className="flex justify-between items-center mt-2">
            <p className="text-xs text-gray-500">
              Press Ctrl+Enter to analyze quickly
            </p>
            <span className="text-xs text-gray-400">
              {text.length} characters
            </span>
          </div>
        </div>

        <button
          onClick={handleAnalyze}
          disabled={!text.trim() || isAnalyzing}
          className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 px-6 rounded-xl font-semibold hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center justify-center space-x-3 shadow-lg hover:shadow-xl"
        >
          {isAnalyzing ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Analyzing with AI Models...</span>
            </>
          ) : (
            <>
              <Send className="h-5 w-5" />
              <span>Analyze Sentiment</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default MainAnalyzer;