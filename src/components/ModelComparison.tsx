import React from 'react';
import { BarChart3, TreePine, Layers, Brain, TrendingUp, Target } from 'lucide-react';

interface ModelComparisonProps {
  analysisHistory: any[];
}

const ModelComparison: React.FC<ModelComparisonProps> = ({ analysisHistory }) => {
  const calculateMetrics = () => {
    if (analysisHistory.length === 0) {
      return {
        randomForest: { accuracy: 0, agreement: 0, avgConfidence: 0 },
        cnn: { accuracy: 0, agreement: 0, avgConfidence: 0 },
        hybrid: { accuracy: 0, avgConfidence: 0 }
      };
    }

    const totalAnalyses = analysisHistory.length;
    let rfAgreement = 0;
    let cnnAgreement = 0;
    let totalRfConfidence = 0;
    let totalCnnConfidence = 0;
    let totalHybridConfidence = 0;

    analysisHistory.forEach(analysis => {
      if (analysis.randomForest.sentiment === analysis.hybrid.sentiment) rfAgreement++;
      if (analysis.cnn.sentiment === analysis.hybrid.sentiment) cnnAgreement++;
      
      totalRfConfidence += analysis.randomForest.confidence;
      totalCnnConfidence += analysis.cnn.confidence;
      totalHybridConfidence += analysis.hybrid.confidence;
    });

    return {
      randomForest: {
        accuracy: 87.5, // Simulated accuracy
        agreement: (rfAgreement / totalAnalyses) * 100,
        avgConfidence: (totalRfConfidence / totalAnalyses) * 100
      },
      cnn: {
        accuracy: 91.2, // Simulated accuracy
        agreement: (cnnAgreement / totalAnalyses) * 100,
        avgConfidence: (totalCnnConfidence / totalAnalyses) * 100
      },
      hybrid: {
        accuracy: 93.8, // Simulated accuracy
        avgConfidence: (totalHybridConfidence / totalAnalyses) * 100
      }
    };
  };

  const metrics = calculateMetrics();

  const modelCards = [
    {
      name: 'Random Forest',
      icon: TreePine,
      color: 'green',
      accuracy: metrics.randomForest.accuracy,
      confidence: metrics.randomForest.avgConfidence,
      agreement: metrics.randomForest.agreement,
      description: 'Traditional ensemble method using decision trees'
    },
    {
      name: 'CNN',
      icon: Layers,
      color: 'blue',
      accuracy: metrics.cnn.accuracy,
      confidence: metrics.cnn.avgConfidence,
      agreement: metrics.cnn.agreement,
      description: 'Deep learning model for text pattern recognition'
    },
    {
      name: 'Hybrid Model',
      icon: Brain,
      color: 'purple',
      accuracy: metrics.hybrid.accuracy,
      confidence: metrics.hybrid.avgConfidence,
      agreement: 100, // Hybrid is always in agreement with itself
      description: 'Weighted ensemble combining both models'
    }
  ];

  const getColorClasses = (color: string) => {
    const colors = {
      green: 'text-green-400 bg-green-500/10 border-green-500/30',
      blue: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
      purple: 'text-purple-400 bg-purple-500/10 border-purple-500/30'
    };
    return colors[color as keyof typeof colors];
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center space-x-3 mb-6">
        <BarChart3 className="h-6 w-6 text-blue-400" />
        <h2 className="text-xl font-semibold text-white">Model Performance Comparison</h2>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Target className="h-5 w-5 text-blue-400" />
            <span className="text-sm font-medium text-gray-300">Total Analyses</span>
          </div>
          <div className="text-2xl font-bold text-white">{analysisHistory.length}</div>
        </div>
        
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            <span className="text-sm font-medium text-gray-300">Best Model</span>
          </div>
          <div className="text-2xl font-bold text-white">Hybrid</div>
          <div className="text-xs text-gray-400">93.8% accuracy</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Brain className="h-5 w-5 text-purple-400" />
            <span className="text-sm font-medium text-gray-300">Model Agreement</span>
          </div>
          <div className="text-2xl font-bold text-white">
            {analysisHistory.length > 0 
              ? Math.round((metrics.randomForest.agreement + metrics.cnn.agreement) / 2)
              : 0}%
          </div>
        </div>
      </div>

      {/* Model Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {modelCards.map((model, index) => {
          const Icon = model.icon;
          return (
            <div key={index} className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6">
              <div className="flex items-center space-x-3 mb-4">
                <div className={`p-2 rounded-lg border ${getColorClasses(model.color)}`}>
                  <Icon className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-semibold text-white">{model.name}</h3>
                  <p className="text-xs text-gray-400">{model.description}</p>
                </div>
              </div>

              <div className="space-y-4">
                {/* Accuracy */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Accuracy</span>
                    <span className="text-white font-medium">{model.accuracy.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 ${
                        model.color === 'green' ? 'bg-green-500' :
                        model.color === 'blue' ? 'bg-blue-500' : 'bg-purple-500'
                      }`}
                      style={{ width: `${model.accuracy}%` }}
                    ></div>
                  </div>
                </div>

                {/* Average Confidence */}
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Avg Confidence</span>
                    <span className="text-white font-medium">{model.confidence.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full transition-all duration-1000 ${
                        model.color === 'green' ? 'bg-green-400' :
                        model.color === 'blue' ? 'bg-blue-400' : 'bg-purple-400'
                      }`}
                      style={{ width: `${model.confidence}%` }}
                    ></div>
                  </div>
                </div>

                {/* Agreement with Hybrid (not shown for hybrid itself) */}
                {model.name !== 'Hybrid Model' && (
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Agreement</span>
                      <span className="text-white font-medium">{model.agreement.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all duration-1000 ${
                          model.color === 'green' ? 'bg-green-300' : 'bg-blue-300'
                        }`}
                        style={{ width: `${model.agreement}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Performance Insights */}
      {analysisHistory.length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Performance Insights</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <h4 className="font-medium text-gray-300">Model Strengths</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Random Forest: Fast inference, interpretable decisions</li>
                <li>• CNN: Excellent pattern recognition, handles complex text</li>
                <li>• Hybrid: Combines strengths, highest overall accuracy</li>
              </ul>
            </div>
            <div className="space-y-2">
              <h4 className="font-medium text-gray-300">Recommendations</h4>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>• Use hybrid model for best accuracy</li>
                <li>• Consider Random Forest for speed-critical applications</li>
                <li>• CNN excels with longer, complex text inputs</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparison;