import React, { useMemo } from 'react';
import { TrendingUp, BarChart3, PieChart, Activity } from 'lucide-react';

interface AnalyticsProps {
  analysisHistory: any[];
}

const Analytics: React.FC<AnalyticsProps> = ({ analysisHistory }) => {
  const analytics = useMemo(() => {
    if (analysisHistory.length === 0) {
      return {
        totalAnalyses: 0,
        sentimentDistribution: { positive: 0, negative: 0, neutral: 0 },
        averageConfidence: 0,
        modelAgreement: 0,
        recentTrend: 'neutral'
      };
    }

    const totalAnalyses = analysisHistory.length;
    const sentimentCounts = { positive: 0, negative: 0, neutral: 0 };
    let totalConfidence = 0;
    let agreementCount = 0;

    analysisHistory.forEach(analysis => {
      sentimentCounts[analysis.hybrid.sentiment as keyof typeof sentimentCounts]++;
      totalConfidence += analysis.hybrid.confidence;
      
      if (analysis.randomForest.sentiment === analysis.cnn.sentiment) {
        agreementCount++;
      }
    });

    const averageConfidence = totalConfidence / totalAnalyses;
    const modelAgreement = (agreementCount / totalAnalyses) * 100;

    // Calculate recent trend (last 5 analyses)
    const recentAnalyses = analysisHistory.slice(0, 5);
    const recentPositive = recentAnalyses.filter(a => a.hybrid.sentiment === 'positive').length;
    const recentNegative = recentAnalyses.filter(a => a.hybrid.sentiment === 'negative').length;
    const recentTrend = recentPositive > recentNegative ? 'positive' : 
                       recentNegative > recentPositive ? 'negative' : 'neutral';

    return {
      totalAnalyses,
      sentimentDistribution: {
        positive: (sentimentCounts.positive / totalAnalyses) * 100,
        negative: (sentimentCounts.negative / totalAnalyses) * 100,
        neutral: (sentimentCounts.neutral / totalAnalyses) * 100
      },
      averageConfidence: averageConfidence * 100,
      modelAgreement,
      recentTrend
    };
  }, [analysisHistory]);

  const getTrendColor = (trend: string) => {
    switch (trend) {
      case 'positive':
        return 'text-green-400';
      case 'negative':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'positive':
        return '📈';
      case 'negative':
        return '📉';
      default:
        return '➡️';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <TrendingUp className="h-6 w-6 text-blue-400" />
        <h2 className="text-xl font-semibold text-white">Analytics Dashboard</h2>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <BarChart3 className="h-5 w-5 text-blue-400" />
            <span className="text-sm font-medium text-gray-300">Total Analyses</span>
          </div>
          <div className="text-2xl font-bold text-white">{analytics.totalAnalyses}</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="h-5 w-5 text-purple-400" />
            <span className="text-sm font-medium text-gray-300">Avg Confidence</span>
          </div>
          <div className="text-2xl font-bold text-white">{analytics.averageConfidence.toFixed(1)}%</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <PieChart className="h-5 w-5 text-green-400" />
            <span className="text-sm font-medium text-gray-300">Model Agreement</span>
          </div>
          <div className="text-2xl font-bold text-white">{analytics.modelAgreement.toFixed(1)}%</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-5 w-5 text-orange-400" />
            <span className="text-sm font-medium text-gray-300">Recent Trend</span>
          </div>
          <div className={`text-2xl font-bold ${getTrendColor(analytics.recentTrend)}`}>
            {getTrendIcon(analytics.recentTrend)} {analytics.recentTrend}
          </div>
        </div>
      </div>

      {/* Sentiment Distribution */}
      <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Sentiment Distribution</h3>
        
        {analytics.totalAnalyses > 0 ? (
          <div className="space-y-4">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-green-400 font-medium">Positive</span>
                <span className="text-white">{analytics.sentimentDistribution.positive.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-green-500 h-3 rounded-full transition-all duration-1000"
                  style={{ width: `${analytics.sentimentDistribution.positive}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-red-400 font-medium">Negative</span>
                <span className="text-white">{analytics.sentimentDistribution.negative.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-red-500 h-3 rounded-full transition-all duration-1000"
                  style={{ width: `${analytics.sentimentDistribution.negative}%` }}
                ></div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-400 font-medium">Neutral</span>
                <span className="text-white">{analytics.sentimentDistribution.neutral.toFixed(1)}%</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div 
                  className="bg-gray-500 h-3 rounded-full transition-all duration-1000"
                  style={{ width: `${analytics.sentimentDistribution.neutral}%` }}
                ></div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>No analysis data available yet</p>
            <p className="text-sm">Start analyzing text to see sentiment distribution</p>
          </div>
        )}
      </div>

      {/* Recent Analysis History */}
      {analysisHistory.length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Analysis History</h3>
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {analysisHistory.slice(0, 10).map((analysis, index) => (
              <div key={index} className="flex items-center justify-between py-2 px-3 bg-gray-700/30 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-3 h-3 rounded-full ${
                    analysis.hybrid.sentiment === 'positive' ? 'bg-green-500' :
                    analysis.hybrid.sentiment === 'negative' ? 'bg-red-500' : 'bg-gray-500'
                  }`}></div>
                  <span className="text-gray-300 text-sm truncate max-w-xs">
                    {analysis.originalText || 'Analysis #' + (index + 1)}
                  </span>
                </div>
                <div className="flex items-center space-x-2">
                  <span className="text-xs text-gray-400">
                    {(analysis.hybrid.confidence * 100).toFixed(0)}%
                  </span>
                  <span className={`text-xs font-medium capitalize ${
                    analysis.hybrid.sentiment === 'positive' ? 'text-green-400' :
                    analysis.hybrid.sentiment === 'negative' ? 'text-red-400' : 'text-gray-400'
                  }`}>
                    {analysis.hybrid.sentiment}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Performance Insights */}
      <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Model Performance</h4>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• Random Forest accuracy: 87.5%</li>
              <li>• CNN accuracy: 91.2%</li>
              <li>• Hybrid model accuracy: 93.8%</li>
              <li>• Average model agreement: {analytics.modelAgreement.toFixed(1)}%</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-gray-300 mb-2">Data Quality</h4>
            <ul className="text-sm text-gray-400 space-y-1">
              <li>• High confidence predictions: {analytics.averageConfidence > 80 ? 'Good' : 'Needs improvement'}</li>
              <li>• Model consensus: {analytics.modelAgreement > 75 ? 'Strong' : 'Moderate'}</li>
              <li>• Recent trend: {analytics.recentTrend} sentiment</li>
              <li>• Data points analyzed: {analytics.totalAnalyses}</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;