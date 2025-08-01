// Simulated sentiment analysis functions
// In a real implementation, these would call actual ML models

interface SentimentResult {
  sentiment: 'positive' | 'negative' | 'neutral';
  confidence: number;
}

interface HybridResult extends SentimentResult {
  scores: {
    positive: number;
    negative: number;
    neutral: number;
  };
  agreement: boolean;
}

interface AnalysisResult {
  randomForest: SentimentResult;
  cnn: SentimentResult;
  hybrid: HybridResult;
  timestamp: string;
}

// Simple keyword-based sentiment analysis for simulation
const positiveWords = [
  'amazing', 'excellent', 'fantastic', 'great', 'wonderful', 'awesome', 'perfect',
  'love', 'best', 'incredible', 'outstanding', 'brilliant', 'superb', 'delicious',
  'excited', 'happy', 'joy', 'beautiful', 'impressive', 'satisfied', 'pleased'
];

const negativeWords = [
  'terrible', 'awful', 'horrible', 'bad', 'worst', 'hate', 'disgusting',
  'disappointing', 'annoying', 'frustrated', 'angry', 'sad', 'boring',
  'useless', 'ridiculous', 'stupid', 'waste', 'poor', 'failed', 'broken'
];

const calculateBaseSentiment = (text: string): { sentiment: 'positive' | 'negative' | 'neutral', score: number } => {
  const words = text.toLowerCase().split(/\W+/);
  let positiveScore = 0;
  let negativeScore = 0;

  words.forEach(word => {
    if (positiveWords.includes(word)) positiveScore++;
    if (negativeWords.includes(word)) negativeScore++;
  });

  // Add some randomness to simulate model variations
  const randomFactor = (Math.random() - 0.5) * 0.3;
  const netScore = (positiveScore - negativeScore) + randomFactor;

  if (netScore > 0.2) {
    return { sentiment: 'positive', score: Math.min(0.95, 0.6 + Math.abs(netScore) * 0.2) };
  } else if (netScore < -0.2) {
    return { sentiment: 'negative', score: Math.min(0.95, 0.6 + Math.abs(netScore) * 0.2) };
  } else {
    return { sentiment: 'neutral', score: Math.max(0.5, 0.8 - Math.abs(netScore) * 0.3) };
  }
};

const simulateRandomForest = (text: string): SentimentResult => {
  const base = calculateBaseSentiment(text);
  
  // Random Forest tends to be more conservative
  const confidence = Math.max(0.4, base.score * 0.9 + (Math.random() * 0.1));
  
  return {
    sentiment: base.sentiment,
    confidence: confidence
  };
};

const simulateCNN = (text: string): SentimentResult => {
  const base = calculateBaseSentiment(text);
  
  // CNN tends to be more confident but can vary more
  const variance = (Math.random() - 0.5) * 0.4;
  let confidence = Math.max(0.3, Math.min(0.98, base.score + variance));
  
  // CNN might sometimes disagree with base sentiment for shorter texts
  let sentiment = base.sentiment;
  if (text.length < 50 && Math.random() > 0.8) {
    const sentiments: ('positive' | 'negative' | 'neutral')[] = ['positive', 'negative', 'neutral'];
    sentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
    confidence *= 0.7; // Lower confidence for disagreement
  }
  
  return {
    sentiment,
    confidence
  };
};

const calculateHybridPrediction = (rf: SentimentResult, cnn: SentimentResult): HybridResult => {
  // Weighted average (CNN gets slightly higher weight due to better accuracy)
  const rfWeight = 0.4;
  const cnnWeight = 0.6;
  
  // Calculate weighted scores for each sentiment
  const scores = {
    positive: 0,
    negative: 0,
    neutral: 0
  };
  
  // Convert individual predictions to score distributions
  const rfScores = { positive: 0, negative: 0, neutral: 0 };
  const cnnScores = { positive: 0, negative: 0, neutral: 0 };
  
  rfScores[rf.sentiment] = rf.confidence;
  rfScores.positive += rf.sentiment !== 'positive' ? (1 - rf.confidence) * 0.3 : 0;
  rfScores.negative += rf.sentiment !== 'negative' ? (1 - rf.confidence) * 0.3 : 0;
  rfScores.neutral += rf.sentiment !== 'neutral' ? (1 - rf.confidence) * 0.4 : 0;
  
  cnnScores[cnn.sentiment] = cnn.confidence;
  cnnScores.positive += cnn.sentiment !== 'positive' ? (1 - cnn.confidence) * 0.3 : 0;
  cnnScores.negative += cnn.sentiment !== 'negative' ? (1 - cnn.confidence) * 0.3 : 0;
  cnnScores.neutral += cnn.sentiment !== 'neutral' ? (1 - cnn.confidence) * 0.4 : 0;
  
  // Combine weighted scores
  scores.positive = rfScores.positive * rfWeight + cnnScores.positive * cnnWeight;
  scores.negative = rfScores.negative * rfWeight + cnnScores.negative * cnnWeight;
  scores.neutral = rfScores.neutral * rfWeight + cnnScores.neutral * cnnWeight;
  
  // Normalize scores
  const total = scores.positive + scores.negative + scores.neutral;
  scores.positive /= total;
  scores.negative /= total;
  scores.neutral /= total;
  
  // Determine final sentiment
  const maxScore = Math.max(scores.positive, scores.negative, scores.neutral);
  let finalSentiment: 'positive' | 'negative' | 'neutral';
  
  if (scores.positive === maxScore) {
    finalSentiment = 'positive';
  } else if (scores.negative === maxScore) {
    finalSentiment = 'negative';
  } else {
    finalSentiment = 'neutral';
  }
  
  const agreement = rf.sentiment === cnn.sentiment;
  const confidence = agreement ? 
    Math.min(0.98, maxScore * 1.1) : // Boost confidence when models agree
    Math.max(0.5, maxScore); // Maintain reasonable confidence when they disagree
  
  return {
    sentiment: finalSentiment,
    confidence,
    scores,
    agreement
  };
};

export const analyzeSentiment = (text: string): AnalysisResult => {
  const randomForest = simulateRandomForest(text);
  const cnn = simulateCNN(text);
  const hybrid = calculateHybridPrediction(randomForest, cnn);
  
  return {
    randomForest,
    cnn,
    hybrid,
    timestamp: new Date().toISOString()
  };
};