import React, { useState } from 'react';
import { Brain, Activity, Database, BarChart2, Sparkles } from 'lucide-react';
import MainAnalyzer from './components/MainAnalyzer';
import ResultsDisplay from './components/ResultsDisplay';
import DataSamples from './components/DataSamples';
import PerformanceMetrics from './components/PerformanceMetrics';

function App() {
  const [currentAnalysis, setCurrentAnalysis] = useState<any>(null);
  const [analysisHistory, setAnalysisHistory] = useState<any[]>([]);

  const handleNewAnalysis = (analysis: any) => {
    setCurrentAnalysis(analysis);
    setAnalysisHistory(prev => [analysis, ...prev.slice(0, 49)]);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-3 rounded-xl shadow-lg">
                <Brain className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">SentiHybrid AI</h1>
                <p className="text-sm text-gray-600">Advanced Sentiment Classification System</p>
              </div>
            </div>
            <div className="flex items-center space-x-2 bg-green-50 px-4 py-2 rounded-full">
              <Activity className="h-4 w-4 text-green-600" />
              <span className="text-sm font-medium text-green-700">Live System</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Input & Analysis */}
          <div className="lg:col-span-2 space-y-6">
            <MainAnalyzer onAnalysisComplete={handleNewAnalysis} />
            {currentAnalysis && <ResultsDisplay analysis={currentAnalysis} />}
          </div>

          {/* Right Column - Sidebar */}
          <div className="space-y-6">
            <DataSamples onSampleAnalyze={handleNewAnalysis} />
            <PerformanceMetrics analysisHistory={analysisHistory} />
          </div>
        </div>

        {/* Bottom Section - Model Comparison */}
        {analysisHistory.length > 0 && (
          <div className="mt-12">
            <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-8">
              <div className="flex items-center space-x-3 mb-6">
                <BarChart2 className="h-6 w-6 text-blue-600" />
                <h2 className="text-xl font-semibold text-gray-900">Model Performance Analysis</h2>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center p-6 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-200">
                  <div className="text-3xl font-bold text-green-700 mb-2">87.5%</div>
                  <div className="text-sm font-medium text-green-600 mb-1">Random Forest</div>
                  <div className="text-xs text-gray-600">Traditional ML Accuracy</div>
                </div>
                
                <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl border border-blue-200">
                  <div className="text-3xl font-bold text-blue-700 mb-2">91.2%</div>
                  <div className="text-sm font-medium text-blue-600 mb-1">CNN Model</div>
                  <div className="text-xs text-gray-600">Deep Learning Accuracy</div>
                </div>
                
                <div className="text-center p-6 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl border border-purple-200">
                  <div className="text-3xl font-bold text-purple-700 mb-2">93.8%</div>
                  <div className="text-sm font-medium text-purple-600 mb-1">Hybrid System</div>
                  <div className="text-xs text-gray-600">Combined Accuracy</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center justify-center space-x-2 text-gray-600">
            <Sparkles className="h-4 w-4" />
            <span className="text-sm">Powered by Advanced Machine Learning</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;