import React, { useState, useCallback } from 'react';
import { Send, Brain, TreePine, Layers, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { analyzeSentiment } from '../utils/sentimentAnalysis';

interface SentimentAnalyzerProps {
  onAnalysisComplete: (analysis: any) => void;
}

const SentimentAnalyzer: React.FC<SentimentAnalyzerProps> = ({ onAnalysisComplete }) => {
  const [text, setText] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleAnalyze = useCallback(async () => {
    if (!text.trim() || isAnalyzing) return;

    setIsAnalyzing(true);
    
    // Simulate analysis delay for realistic UX
    setTimeout(() => {
      const analysis = analyzeSentiment(text);
      setResult(analysis);
      onAnalysisComplete(analysis);
      setIsAnalyzing(false);
    }, 1500);
  }, [text, isAnalyzing, onAnalysisComplete]);

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <TrendingUp className="h-5 w-5 text-green-400" />;
      case 'negative':
        return <TrendingDown className="h-5 w-5 text-red-400" />;
      default:
        return <Minus className="h-5 w-5 text-gray-400" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-400 bg-green-500/10 border-green-500/30';
      case 'negative':
        return 'text-red-400 bg-red-500/10 border-red-500/30';
      default:
        return 'text-gray-400 bg-gray-500/10 border-gray-500/30';
    }
  };

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Analyze Sentiment</h2>
        <div className="space-y-4">
          <div>
            <label htmlFor="text-input" className="block text-sm font-medium text-gray-300 mb-2">
              Enter text to analyze
            </label>
            <textarea
              id="text-input"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Type or paste your text here... (e.g., tweets, reviews, comments)"
              rows={4}
              className="w-full bg-gray-900/50 border border-gray-600/50 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 transition-all duration-200"
            />
          </div>
          <button
            onClick={handleAnalyze}
            disabled={!text.trim() || isAnalyzing}
            className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          >
            {isAnalyzing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                <span>Analyzing...</span>
              </>
            ) : (
              <>
                <Send className="h-4 w-4" />
                <span>Analyze Sentiment</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Individual Model Results */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Individual Models</h3>
            
            {/* Random Forest */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
              <div className="flex items-center space-x-3 mb-3">
                <TreePine className="h-5 w-5 text-green-400" />
                <h4 className="font-medium text-white">Random Forest</h4>
              </div>
              <div className="space-y-2">
                <div className={`inline-block px-3 py-1 rounded-full border text-sm font-medium ${getSentimentColor(result.randomForest.sentiment)}`}>
                  {getSentimentIcon(result.randomForest.sentiment)}
                  <span className="ml-2 capitalize">{result.randomForest.sentiment}</span>
                </div>
                <div className="text-sm text-gray-400">
                  Confidence: {(result.randomForest.confidence * 100).toFixed(1)}%
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${result.randomForest.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* CNN */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
              <div className="flex items-center space-x-3 mb-3">
                <Layers className="h-5 w-5 text-blue-400" />
                <h4 className="font-medium text-white">CNN</h4>
              </div>
              <div className="space-y-2">
                <div className={`inline-block px-3 py-1 rounded-full border text-sm font-medium ${getSentimentColor(result.cnn.sentiment)}`}>
                  {getSentimentIcon(result.cnn.sentiment)}
                  <span className="ml-2 capitalize">{result.cnn.sentiment}</span>
                </div>
                <div className="text-sm text-gray-400">
                  Confidence: {(result.cnn.confidence * 100).toFixed(1)}%
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${result.cnn.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>
          </div>

          {/* Hybrid Result */}
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-white">Hybrid Prediction</h3>
            
            <div className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6">
              <div className="flex items-center space-x-3 mb-4">
                <Brain className="h-6 w-6 text-purple-400" />
                <h4 className="text-lg font-medium text-white">Ensemble Result</h4>
              </div>
              
              <div className="space-y-4">
                <div className={`inline-flex items-center px-4 py-2 rounded-lg border text-lg font-semibold ${getSentimentColor(result.hybrid.sentiment)}`}>
                  {getSentimentIcon(result.hybrid.sentiment)}
                  <span className="ml-2 capitalize">{result.hybrid.sentiment}</span>
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Overall Confidence</span>
                    <span className="text-white font-medium">{(result.hybrid.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-1000"
                      style={{ width: `${result.hybrid.confidence * 100}%` }}
                    ></div>
                  </div>
                </div>

                {result.hybrid.agreement && (
                  <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-3">
                    <div className="text-green-400 text-sm font-medium">
                      ✓ Models are in agreement
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Detailed Scores */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
              <h4 className="font-medium text-white mb-3">Detailed Scores</h4>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">Positive:</span>
                  <span className="text-green-400">{(result.hybrid.scores.positive * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Negative:</span>
                  <span className="text-red-400">{(result.hybrid.scores.negative * 100).toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Neutral:</span>
                  <span className="text-gray-400">{(result.hybrid.scores.neutral * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default SentimentAnalyzer;