import React from 'react';
import { TreePine, Layers, Brain, TrendingUp, TrendingDown, Minus, CheckCircle, AlertCircle } from 'lucide-react';

interface ResultsDisplayProps {
  analysis: any;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ analysis }) => {
  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <TrendingUp className="h-5 w-5 text-green-600" />;
      case 'negative':
        return <TrendingDown className="h-5 w-5 text-red-600" />;
      default:
        return <Minus className="h-5 w-5 text-gray-600" />;
    }
  };

  const getSentimentBadge = (sentiment: string) => {
    const baseClasses = "inline-flex items-center px-3 py-1 rounded-full text-sm font-medium";
    switch (sentiment) {
      case 'positive':
        return `${baseClasses} bg-green-100 text-green-800 border border-green-200`;
      case 'negative':
        return `${baseClasses} bg-red-100 text-red-800 border border-red-200`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800 border border-gray-200`;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'bg-green-500';
    if (confidence > 0.6) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
        <div className="flex items-center space-x-3 mb-6">
          <Brain className="h-6 w-6 text-purple-600" />
          <h3 className="text-xl font-semibold text-gray-900">Hybrid Prediction</h3>
          {analysis.hybrid.agreement && (
            <div className="flex items-center space-x-1 bg-green-50 px-3 py-1 rounded-full">
              <CheckCircle className="h-4 w-4 text-green-600" />
              <span className="text-xs font-medium text-green-700">Models Agree</span>
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="space-y-4">
            <div className={getSentimentBadge(analysis.hybrid.sentiment)}>
              {getSentimentIcon(analysis.hybrid.sentiment)}
              <span className="ml-2 capitalize">{analysis.hybrid.sentiment}</span>
            </div>
            
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Confidence Level</span>
                <span className="font-semibold text-gray-900">
                  {(analysis.hybrid.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3">
                <div 
                  className={`h-3 rounded-full transition-all duration-1000 ${getConfidenceColor(analysis.hybrid.confidence)}`}
                  style={{ width: `${analysis.hybrid.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          <div className="space-y-3">
            <h4 className="font-medium text-gray-700">Detailed Scores</h4>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Positive</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                      style={{ width: `${analysis.hybrid.scores.positive * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {(analysis.hybrid.scores.positive * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Negative</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-red-500 h-2 rounded-full transition-all duration-1000"
                      style={{ width: `${analysis.hybrid.scores.negative * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {(analysis.hybrid.scores.negative * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Neutral</span>
                <div className="flex items-center space-x-2">
                  <div className="w-20 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-gray-500 h-2 rounded-full transition-all duration-1000"
                      style={{ width: `${analysis.hybrid.scores.neutral * 100}%` }}
                    ></div>
                  </div>
                  <span className="text-sm font-medium text-gray-900 w-12">
                    {(analysis.hybrid.scores.neutral * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Individual Models */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <TreePine className="h-5 w-5 text-green-600" />
            <h4 className="font-semibold text-gray-900">Random Forest</h4>
          </div>
          
          <div className="space-y-3">
            <div className={getSentimentBadge(analysis.randomForest.sentiment)}>
              {getSentimentIcon(analysis.randomForest.sentiment)}
              <span className="ml-2 capitalize">{analysis.randomForest.sentiment}</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Confidence</span>
                <span className="font-medium text-gray-900">
                  {(analysis.randomForest.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                  style={{ width: `${analysis.randomForest.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-md border border-gray-200 p-6">
          <div className="flex items-center space-x-3 mb-4">
            <Layers className="h-5 w-5 text-blue-600" />
            <h4 className="font-semibold text-gray-900">CNN Model</h4>
          </div>
          
          <div className="space-y-3">
            <div className={getSentimentBadge(analysis.cnn.sentiment)}>
              {getSentimentIcon(analysis.cnn.sentiment)}
              <span className="ml-2 capitalize">{analysis.cnn.sentiment}</span>
            </div>
            
            <div className="space-y-1">
              <div className="flex justify-between text-sm">
                <span className="text-gray-600">Confidence</span>
                <span className="font-medium text-gray-900">
                  {(analysis.cnn.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-1000"
                  style={{ width: `${analysis.cnn.confidence * 100}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ResultsDisplay;