import React, { useState } from 'react';
import { Database, Play, Loader2 } from 'lucide-react';
import { analyzeSentiment } from '../utils/sentimentAnalysis';

interface DataSamplesProps {
  onSampleAnalyze: (analysis: any) => void;
}

const DataSamples: React.FC<DataSamplesProps> = ({ onSampleAnalyze }) => {
  const [isAnalyzing, setIsAnalyzing] = useState<number | null>(null);

  const samples = [
    {
      id: 1,
      text: "Just had the most amazing experience at the new restaurant! The food was incredible! 🍕✨",
      category: "Restaurant Review"
    },
    {
      id: 2,
      text: "This movie was a complete waste of time. Poor acting and terrible plot.",
      category: "Movie Review"
    },
    {
      id: 3,
      text: "The weather is okay today. Not too hot, not too cold.",
      category: "Weather"
    },
    {
      id: 4,
      text: "Excited about the new product launch! This could be a game-changer! 🚀",
      category: "Product News"
    }
  ];

  const handleAnalyzeSample = async (sample: any) => {
    setIsAnalyzing(sample.id);
    
    setTimeout(() => {
      const analysis = {
        ...analyzeSentiment(sample.text),
        originalText: sample.text,
        category: sample.category,
        timestamp: new Date().toISOString()
      };
      
      onSampleAnalyze(analysis);
      setIsAnalyzing(null);
    }, 1500);
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg border border-gray-200 p-6">
      <div className="flex items-center space-x-3 mb-6">
        <Database className="h-5 w-5 text-blue-600" />
        <h3 className="text-lg font-semibold text-gray-900">Sample Data</h3>
      </div>

      <div className="space-y-4">
        {samples.map((sample) => (
          <div key={sample.id} className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors duration-200">
            <div className="mb-2">
              <span className="inline-block bg-blue-50 text-blue-700 text-xs font-medium px-2 py-1 rounded-full">
                {sample.category}
              </span>
            </div>
            
            <p className="text-sm text-gray-700 mb-3 leading-relaxed">
              {sample.text}
            </p>
            
            <button
              onClick={() => handleAnalyzeSample(sample)}
              disabled={isAnalyzing !== null}
              className="w-full bg-blue-50 hover:bg-blue-100 text-blue-700 py-2 px-3 rounded-lg text-sm font-medium transition-colors duration-200 flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              {isAnalyzing === sample.id ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  <span>Test Sample</span>
                </>
              )}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DataSamples;