import React, { useState } from 'react';
import { Target, Play, MessageSquare, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { analyzeSentiment } from '../utils/sentimentAnalysis';

interface SampleDataProps {
  onAnalyze: (analysis: any) => void;
}

const SampleData: React.FC<SampleDataProps> = ({ onAnalyze }) => {
  const [selectedSample, setSelectedSample] = useState<number | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const sampleTweets = [
    {
      id: 1,
      text: "Just had the most amazing experience at the new restaurant downtown! The service was incredible and the food was absolutely delicious. Definitely coming back! 🍕✨",
      expectedSentiment: "positive",
      category: "Restaurant Review"
    },
    {
      id: 2,
      text: "This movie was a complete waste of time. Poor acting, terrible plot, and the ending made no sense whatsoever. Would not recommend to anyone.",
      expectedSentiment: "negative",
      category: "Movie Review"
    },
    {
      id: 3,
      text: "The weather is okay today. Not too hot, not too cold. Just average I guess.",
      expectedSentiment: "neutral",
      category: "Weather Comment"
    },
    {
      id: 4,
      text: "I'm so excited about the new product launch! This could be a game-changer for the industry. Can't wait to see what happens next! 🚀",
      expectedSentiment: "positive",
      category: "Product Launch"
    },
    {
      id: 5,
      text: "Stuck in traffic again... This commute is getting worse every day. Why can't they fix these roads?",
      expectedSentiment: "negative",
      category: "Traffic Complaint"
    },
    {
      id: 6,
      text: "The conference was informative. Some good speakers, some not so much. Overall it was fine.",
      expectedSentiment: "neutral",
      category: "Event Review"
    },
    {
      id: 7,
      text: "Best vacation ever! The beaches were pristine, the people were friendly, and the food was amazing. Already planning my next trip back! 🏖️😍",
      expectedSentiment: "positive",
      category: "Travel Review"
    },
    {
      id: 8,
      text: "Customer service was absolutely terrible. Waited on hold for 2 hours just to be told they can't help me. Switching to a different company.",
      expectedSentiment: "negative",
      category: "Customer Service"
    }
  ];

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <TrendingUp className="h-4 w-4 text-green-400" />;
      case 'negative':
        return <TrendingDown className="h-4 w-4 text-red-400" />;
      default:
        return <Minus className="h-4 w-4 text-gray-400" />;
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

  const handleAnalyzeSample = async (sample: any) => {
    setSelectedSample(sample.id);
    setIsAnalyzing(true);
    
    // Simulate analysis delay
    setTimeout(() => {
      const analysis = {
        ...analyzeSentiment(sample.text),
        originalText: sample.text,
        expectedSentiment: sample.expectedSentiment,
        category: sample.category,
        timestamp: new Date().toISOString()
      };
      
      onAnalyze(analysis);
      setIsAnalyzing(false);
      setSelectedSample(null);
    }, 1500);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <Target className="h-6 w-6 text-blue-400" />
        <h2 className="text-xl font-semibold text-white">Sample Twitter Data</h2>
      </div>

      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6 mb-6">
        <p className="text-gray-300 text-sm">
          Test the hybrid sentiment analysis system with these sample social media posts. 
          Each sample includes the expected sentiment for comparison with model predictions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {sampleTweets.map((sample) => (
          <div key={sample.id} className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4 hover:border-gray-600/50 transition-all duration-200">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-2">
                <MessageSquare className="h-4 w-4 text-blue-400" />
                <span className="text-xs text-gray-400 font-medium">{sample.category}</span>
              </div>
              <div className={`inline-flex items-center px-2 py-1 rounded-full border text-xs font-medium ${getSentimentColor(sample.expectedSentiment)}`}>
                {getSentimentIcon(sample.expectedSentiment)}
                <span className="ml-1 capitalize">{sample.expectedSentiment}</span>
              </div>
            </div>
            
            <p className="text-gray-300 text-sm mb-4 leading-relaxed">
              {sample.text}
            </p>
            
            <button
              onClick={() => handleAnalyzeSample(sample)}
              disabled={isAnalyzing}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white px-4 py-2 rounded-lg text-sm font-medium hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 w-full justify-center"
            >
              {isAnalyzing && selectedSample === sample.id ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Analyze This Sample</span>
                </>
              )}
            </button>
          </div>
        ))}
      </div>

      <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-2">Dataset Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Total Samples:</span>
            <div className="text-white font-medium">{sampleTweets.length}</div>
          </div>
          <div>
            <span className="text-gray-400">Categories:</span>
            <div className="text-white font-medium">{new Set(sampleTweets.map(s => s.category)).size}</div>
          </div>
          <div>
            <span className="text-gray-400">Sentiment Distribution:</span>
            <div className="text-white font-medium">
              {Math.round((sampleTweets.filter(s => s.expectedSentiment === 'positive').length / sampleTweets.length) * 100)}% Positive
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SampleData;