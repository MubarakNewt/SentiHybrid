import React from 'react';
import { BarChart2, TrendingUp, Target, Activity } from 'lucide-react';

interface PerformanceMetricsProps {
  analysisHistory: any[];
}

const PerformanceMetrics: React.FC<PerformanceMetricsProps> = ({ analysisHistory }) => {
  const calculateStats = () => {
    if (analysisHistory.length === 0) {
      return {
        totalAnalyses: 0,
        avgConfidence: 0,
        sentimentDistribution: { positive: 0, negative: 0, neutral: 0 },
        modelAgreement: 0
      };
    }

    const total = analysisHistory.length;
    let totalConfidence = 0;
    let agreements = 0;
    const sentiments = { positive: 0, negative: 0, neutral: 0 };

    analysisHistory.forEach(analysis => {
      totalConfidence += analysis.hybrid.confidence;
      sentiments[analysis.hybrid.sentiment as keyof typeof sentiments]++;
      
      if (analysis.randomForest.sentiment === analysis.cnn.sentiment) {
        agreements++;
      }
    });

    return {
      totalAnalyses: total,
      avgConfidence: (totalConfidence / total) * 100,
      sentimentDistribution: {
        positive: (sentiments.positive / total) * 100,
        negative: (sentiments.negative / total) * 100,
        neutral: (sentiments.neutral / total) * 100
      },
      modelAgreement: (agreements / total) * 100
    };
  };

  const stats = calculateStats();

  return (
    <div className="space-y-6">
      <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
        <div className="flex items-center space-x-3 mb-6">
          <BarChart2 className="h-5 w-5 text-blue-600" />
          <h3 className="text-lg font-semibold text-gray-900">Performance</h3>
        </div>

        <div className="space-y-4">
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Target className="h-4 w-4 text-blue-600" />
              <span className="text-sm font-medium text-gray-700">Total Analyses</span>
            </div>
            <span className="text-lg font-bold text-gray-900">{stats.totalAnalyses}</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-green-600" />
              <span className="text-sm font-medium text-gray-700">Avg Confidence</span>
            </div>
            <span className="text-lg font-bold text-gray-900">{stats.avgConfidence.toFixed(1)}%</span>
          </div>

          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <TrendingUp className="h-4 w-4 text-purple-600" />
              <span className="text-sm font-medium text-gray-700">Model Agreement</span>
            </div>
            <span className="text-lg font-bold text-gray-900">{stats.modelAgreement.toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {stats.totalAnalyses > 0 && (
        <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
          <h4 className="font-semibold text-gray-900 mb-4">Sentiment Distribution</h4>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Positive</span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-green-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${stats.sentimentDistribution.positive}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-gray-900 w-10">
                  {stats.sentimentDistribution.positive.toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Negative</span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-red-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${stats.sentimentDistribution.negative}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-gray-900 w-10">
                  {stats.sentimentDistribution.negative.toFixed(0)}%
                </span>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Neutral</span>
              <div className="flex items-center space-x-2">
                <div className="w-16 bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gray-500 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${stats.sentimentDistribution.neutral}%` }}
                  ></div>
                </div>
                <span className="text-sm font-medium text-gray-900 w-10">
                  {stats.sentimentDistribution.neutral.toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceMetrics;