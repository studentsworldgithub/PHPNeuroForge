<?php
// core/Utilities.php
// Auther: Majdi M. S. Awad
// Year: 2024
// Version: 1.0.0
// MIT License

/**
 * Matrix Multiplication
 * Performs matrix multiplication.
 *
 * @param array $matrix1    First matrix.
 * @param array $matrix2    Second matrix.
 * @return array            Resultant matrix after multiplication.
 */
function matrixMultiply($matrix1, $matrix2) {
    // Matrix multiplication implementation
}

/**
 * Data Preprocessing Class
 * Contains methods for data preprocessing.
 */
class DataPreprocessing {
    // Data preprocessing methods
}

// utilities.php

class Utilities {
    /**
     * calculateFutureValue
     * Calculates the Future Value of an Ordinary Annuity.
     */
    public function calculateFutureValue($payment, $rate, $periods) {
        // Calculate Future Value (FV) of the annuity
        $interestFactor = pow(1 + $interestRate, $periods);
        return $payment * (($interestFactor - 1) / $interestRate);
    }

    /**
     * calculatePresentValue
     * Calculates the Present Value of an Ordinary Annuity.
     */
    public function calculatePresentValue($payment, $rate, $periods) {
        // Calculate Present Value (PV) of the annuity
        $interestFactor = pow(1 + $interestRate, $periods);
        return $payment * ((1 - (1 / $interestFactor)) / $interestRate);
    }

    /**
     * calculateFutureValueDue
     * Calculates the Future Value of an Annuity Due.
     */
    public function calculateFutureValueDue($payment, $rate, $periods) {
        // Calculate Future Value (FV) of the annuity due
        $interestFactor = pow(1 + $interestRate, $periods);
        return $payment * ((($interestFactor - 1) / $interestRate) * (1 + $interestRate));
    }

    /**
     * calculatePresentValueDue
     * Calculates the Present Value of an Annuity Due.
     */
    public function calculatePresentValueDue($payment, $rate, $periods) {
        $discountFactor = (1 - pow(1 + $interestRate, -$periods)) / $interestRate;
        $presentValueDue = $payment * $discountFactor;
        return $presentValueDue;
    }

    /**
     * calculatePayment
     * Calculates the Payment for an Ordinary Annuity.
     */
    public function calculatePayment($presentValue, $rate, $periods) {
        $payment = ($presentValue * $interestRate) / (1 - pow((1 + $interestRate), -$periods));
        return $payment;
    }
	
    /**
     * calculateNumberOfPeriods
     * Calculates the Number of Periods for an Ordinary Annuity.
     */
    public function calculateNumberOfPeriods($presentValue, $rate, $payment) {
        // Implementation of the formula to calculate Number of Periods (n)
        // Formula: n = ln((P * r) / (P - p * r)) / ln(1 + r)
        
        // Calculate the Number of Periods
        // Example calculation:
        $numerator = log(($presentValue * $rate) / ($presentValue - $payment * $rate));
        $denominator = log(1 + $rate);
        $numPeriods = $numerator / $denominator;

        return $numPeriods;
    }
	
    /**
     * calculatePVFromAnnuityDue
     * Converts Annuity Due Present Value to Ordinary Annuity Present Value.
     */
    public function calculatePVFromAnnuityDue($annuityDuePV) {
        // Implementation of the formula to convert Annuity Due PV to Ordinary PV
        // Formula: PV ordinary = Annuity Due PV * (1 + r)

        // Calculate the Ordinary Present Value
        // Example calculation:
        $ordinaryPV = $annuityDuePV * (1 + $rate);

        return $ordinaryPV;
    }

    /**
     * calculateAnnuityPaymentGrowthRate
     * Calculates the Annuity Payment Growth Rate (g).
     */
    public function calculateAnnuityPaymentGrowthRate($presentValue, $futureValue, $periods) {
        // Implementation of the formula to calculate Annuity Payment Growth Rate (g)
        // Formula: g = ((FV / PV) ^ (1 / n)) - 1

        // Calculate the Annuity Payment Growth Rate
        // Example calculation:
        $growthRate = pow(($futureValue / $presentValue), (1 / $periods)) - 1;

        return $growthRate;
    }
	
    /**
     * calculatePresentValuePerpetuity
     * Calculates the Present Value of Perpetuity (PV perpetuity).
     */
    public function calculatePresentValuePerpetuity($payment, $discountRate) {
        // Calculate the Present Value of Perpetuity
        // Formula: PV = Payment / Discount Rate

        // Calculate the Present Value of Perpetuity
        $presentValue = $payment / $discountRate;

        return $presentValue;
    }	
	
    /**
     * calculateContinuousPV
     * Calculates the Present Value (PV) of an annuity with continuous compounding.
     */
    public function calculateContinuousPV($payment, $interestRate, $periods) {
        // Calculate the Present Value (PV) with continuous compounding
        // Formula: PV = Payment * (1 - exp(-Interest Rate * Periods)) / Interest Rate

        // Calculate the Present Value (PV) with continuous compounding
        $presentValue = $payment * (1 - exp(-$interestRate * $periods)) / $interestRate;

        return $presentValue;
    }	
	
    /**
     * calculateContinuousFV
     * Calculates the Future Value (FV) of an annuity with continuous compounding.
     */
    public function calculateContinuousFV($payment, $interestRate, $periods) {
        // Calculate the Future Value (FV) with continuous compounding
        // Formula: FV = Payment * ((exp(Interest Rate * Periods) - 1) / Interest Rate)

        // Calculate the Future Value (FV) with continuous compounding
        $futureValue = $payment * ((exp($interestRate * $periods) - 1) / $interestRate);

        return $futureValue;
    }

    /**
     * calculateAnnuityDuePayment
     * Calculates the Annuity Due Payment for Present Value.
     */
    public function calculateAnnuityDuePayment($presentValue, $interestRate, $periods) {
        // Calculate the Annuity Due Payment for Present Value
        // Formula: Payment = PV * Interest Rate / (1 - (1 + Interest Rate)^-Periods)

        // Calculate the Annuity Due Payment for Present Value
        $annuityDuePayment = $presentValue * $interestRate / (1 - pow(1 + $interestRate, -$periods));

        return $annuityDuePayment;
    }

    /**
     * calculateEffectiveRate
     * Calculates the Effective Rate for Growth with Different Frequencies.
     */
    public function calculateEffectiveRate($annualRate, $paymentFrequency, $compoundingFrequency) {
        // Calculate the Effective Rate for Growth with Different Frequencies
        // Formula: r_effective = (1 + (annualRate / paymentFrequency)) ^ compoundingFrequency - 1

        $rEffective = pow(1 + ($annualRate / $paymentFrequency), $compoundingFrequency) - 1;

        return $rEffective;
    }

    /**
     * calculatePresentValueGrowingAnnuity
     * Calculates the Present Value of a Growing Annuity.
     */
    public function calculatePresentValueGrowingAnnuity($payment, $growthRate, $discountRate, $nPeriods) {
        // Calculate the Present Value of a Growing Annuity
        // Formula: PV_growing = payment * ((1 - (1 + growthRate) ^ -nPeriods) / (discountRate - growthRate))

        $presentValueGrowingAnnuity = $payment * ((1 - pow(1 + $growthRate, -$nPeriods)) / ($discountRate - $growthRate));

        return $presentValueGrowingAnnuity;
    }

    /**
     * calculateFutureValueGrowingAnnuity
     * Calculates the Future Value of a Growing Annuity.
     */
    public function calculateFutureValueGrowingAnnuity($payment, $growthRate, $discountRate, $nPeriods) {
        // Calculate the Future Value of a Growing Annuity
        // Formula: FV_growing = payment * (((1 + growthRate) ^ nPeriods - (1 + discountRate) ^ nPeriods) / (growthRate - discountRate))

        $futureValueGrowingAnnuity = $payment * (((pow(1 + $growthRate, $nPeriods) - pow(1 + $discountRate, $nPeriods))) / ($growthRate - $discountRate));

        return $futureValueGrowingAnnuity;
    }

    /**
     * calculatePresentValueContinuousAnnuity
     * Calculates the Present Value of an Annuity with Continuous Payments.
     */
    public function calculatePresentValueContinuousAnnuity($payment, $discountRate, $nPeriods) {
        // Calculate the Present Value of an Annuity with Continuous Payments
        // Formula: PV_continuous = payment * (1 - exp(-discountRate * nPeriods)) / discountRate

        $presentValueContinuousAnnuity = $payment * (1 - exp(-$discountRate * $nPeriods)) / $discountRate;

        return $presentValueContinuousAnnuity;
    }

    /**
     * calculatePresentValueVaryingAnnuity
     * Calculates the Present Value of an Annuity with Varying Payments.
     */
    public function calculatePresentValueVaryingAnnuity($payments, $discountRate) {
        // Calculate the Present Value of an Annuity with Varying Payments
        // Formula: PV_varying = Σ(payment[i] / (1 + discountRate)^i), where i ranges from 1 to n

        $presentValueVaryingAnnuity = 0;
        $nPeriods = count($payments);

        for ($i = 0; $i < $nPeriods; $i++) {
            $presentValueVaryingAnnuity += $payments[$i] / pow((1 + $discountRate), $i + 1);
        }

        return $presentValueVaryingAnnuity;
    }

    /**
     * calculateNumberOfPeriodsGrowingAnnuity
     * Calculates the Number of Periods for a Growing Annuity.
     */
    public function calculateNumberOfPeriodsGrowingAnnuity($presentValue, $payment, $growthRate, $discountRate) {
        // Calculate the Number of Periods for a Growing Annuity
        // Formula: n = ln((payment - growthRate * presentValue) / (payment * discountRate - growthRate * presentValue)) / ln(1 + discountRate)

        $numerator = log(($payment - $growthRate * $presentValue) / ($payment * $discountRate - $growthRate * $presentValue));
        $denominator = log(1 + $discountRate);

        $numberOfPeriods = $numerator / $denominator;

        return $numberOfPeriods;
    }

    /**
     * calculateLinearEquation
     * Calculates the Linear Equation (Neuron input).
     */
    public function calculateLinearEquation($weights, $inputs, $bias) {
        // Calculate the Linear Equation
        // Formula: z = (weight1 * input1) + (weight2 * input2) + ... + (weightN * inputN) + bias

        $z = $bias;
        foreach ($weights as $index => $weight) {
            $z += $weight * $inputs[$index];
        }

        return $z;
    }

    /**
     * calculateSigmoid
     * Calculates the Sigmoid function.
     */
    public function calculateSigmoid($z) {
        // Calculate the Sigmoid function
        // Formula: σ(z) = 1 / (1 + exp(-z))

        $sigmoid = 1 / (1 + exp(-$z));

        return $sigmoid;
    }

    /**
     * calculateReLU
     * Calculates the ReLU function.
     */
    public function calculateReLU($z) {
        // Calculate the ReLU function
        // Formula: f(z) = max(0, $z)

        $ReLU = max(0, $z);

        return $ReLU;
    }

    /**
     * calculateTanH
     * Calculates the Hyperbolic Tangent (TanH) function.
     */
    public function calculateTanH($z) {
        // Calculate the TanH function
        // Formula: TanH(z) = (exp($z) - exp(-$z)) / (exp($z) + exp(-$z))

        $TanH = (exp($z) - exp(-$z)) / (exp($z) + exp(-$z));

        return $TanH;
    }

    /**
     * calculateSoftmax
     * Calculates the Softmax function for multi-class classification.
     */
    public function calculateSoftmax($values) {
        // Calculate the Softmax function
        // Formula: Softmax(z_i) = exp($values[$i]) / sum(exp($values))

        $softmaxValues = [];
        $expSum = 0;

        foreach ($values as $val) {
            $expSum += exp($val);
        }

        foreach ($values as $val) {
            $softmaxValues[] = exp($val) / $expSum;
        }

        return $softmaxValues;
    }

    /**
     * calculateMSE
     * Calculates the Mean Squared Error (MSE).
     */
    public function calculateMSE($predictions, $targets) {
        // Calculate the Mean Squared Error (MSE)
        // Formula: MSE = sum((predictions - targets)^2) / n

        $mse = 0;
        $n = count($predictions);

        for ($i = 0; $i < $n; $i++) {
            $mse += pow(($predictions[$i] - $targets[$i]), 2);
        }

        return $mse / $n;
    }

    /**
     * calculateBinaryCrossEntropy
     * Calculates the Binary Cross-Entropy Loss.
     */
    public function calculateBinaryCrossEntropy($predictions, $targets) {
        // Calculate Binary Cross-Entropy Loss
        // Formula: BCE = -sum(targets * log(predictions) + (1 - targets) * log(1 - predictions)) / n

        $bce = 0;
        $n = count($predictions);

        for ($i = 0; $i < $n; $i++) {
            $bce += (-($targets[$i] * log($predictions[$i])) - ((1 - $targets[$i]) * log(1 - $predictions[$i])));
        }

        return -$bce / $n;
    }

    /**
     * calculateMultiClassCrossEntropy
     * Calculates the Multi-class Cross-Entropy Loss.
     */
    public function calculateMultiClassCrossEntropy($predictions, $targets) {
        // Calculate Multi-class Cross-Entropy Loss
        // Formula: CE = -sum(targets * log(predictions)) / n

        $ce = 0;
        $n = count($predictions);

        for ($i = 0; $i < $n; $i++) {
            $ce += -($targets[$i] * log($predictions[$i]));
        }

        return $ce / $n;
    }

    /**
     * gradientDescent
     * Performs Gradient Descent optimization.
     */
    public function gradientDescent($weights, $learningRate, $gradient) {
        // Update weights using Gradient Descent
        // Formula: new_weights = old_weights - learning_rate * gradient

        $updatedWeights = [];
        $numWeights = count($weights);

        for ($i = 0; $i < $numWeights; $i++) {
            $updatedWeights[$i] = $weights[$i] - ($learningRate * $gradient[$i]);
        }

        return $updatedWeights;
    }

    /**
     * chainRule
     * Performs the Chain Rule calculation for derivatives of composite functions.
     */
    public function chainRule($outerFunctionDerivative, $innerFunctionDerivative) {
        // Compute the derivative using the Chain Rule
        // Formula: result = outerFunctionDerivative * innerFunctionDerivative
        
        return $outerFunctionDerivative * $innerFunctionDerivative;
    }

    /**
     * weightUpdate
     * Updates the weights using gradient descent.
     */
    public function weightUpdate($currentWeight, $learningRate, $gradient) {
        // Compute the updated weight using gradient descent
        // Formula: updatedWeight = currentWeight - (learningRate * gradient)
        
        return $currentWeight - ($learningRate * $gradient);
    }

    /**
     * calculateL1Regularization
     * Calculates L1 regularization.
     */
    public function calculateL1Regularization($lambda, $weights) {
        // Compute L1 regularization term
        $l1 = 0;
        foreach ($weights as $weight) {
            $l1 += abs($weight);
        }
        $l1 *= $lambda;

        return $l1;
    }

    /**
     * calculateL2Regularization
     * Calculates L2 regularization.
     */
    public function calculateL2Regularization($lambda, $weights) {
        // Compute L2 regularization term
        $l2 = 0;
        foreach ($weights as $weight) {
            $l2 += $weight ** 2;
        }
        $l2 *= $lambda;

        return $l2;
    }

    /**
     * calculateConvolutionOperation
     * Performs the convolution operation between an input matrix and a filter/kernel.
     */
    public function calculateConvolutionOperation($inputMatrix, $filter) {
        $outputSizeX = count($inputMatrix) - count($filter) + 1;
        $outputSizeY = count($inputMatrix[0]) - count($filter[0]) + 1;

        $outputMatrix = [];

        for ($i = 0; $i < $outputSizeX; $i++) {
            for ($j = 0; $j < $outputSizeY; $j++) {
                $sum = 0;
                for ($m = 0; $m < count($filter); $m++) {
                    for ($n = 0; $n < count($filter[0]); $n++) {
                        $sum += $inputMatrix[$i + $m][$j + $n] * $filter[$m][$n];
                    }
                }
                $outputMatrix[$i][$j] = $sum;
            }
        }

        return $outputMatrix;
    }

    /**
     * calculatePooling
     * Performs max or average pooling on an input matrix or feature map.
     */
    public function calculatePooling($inputMatrix, $windowSize, $poolingType = 'max') {
        $outputSizeX = count($inputMatrix) - $windowSize + 1;
        $outputSizeY = count($inputMatrix[0]) - $windowSize + 1;

        $outputMatrix = [];

        for ($i = 0; $i < $outputSizeX; $i++) {
            for ($j = 0; $j < $outputSizeY; $j++) {
                $values = [];
                for ($m = 0; $m < $windowSize; $m++) {
                    for ($n = 0; $n < $windowSize; $n++) {
                        $values[] = $inputMatrix[$i + $m][$j + $n];
                    }
                }
                if ($poolingType === 'max') {
                    $outputMatrix[$i][$j] = max($values);
                } elseif ($poolingType === 'average') {
                    $outputMatrix[$i][$j] = array_sum($values) / count($values);
                }
            }
        }

        return $outputMatrix;
    }

    /**
     * calculateHiddenStateUpdate
     * Updates the hidden state in a recurrent neural network.
     */
    public function calculateHiddenStateUpdate($inputVector, $prevHiddenState, $weightsInput, $weightsHidden, $bias, $activation = 'tanh') {
        // Assuming inputVector, prevHiddenState, weightsInput, weightsHidden, and bias are arrays/matrices

        $inputWeighted = $this->matrixMultiply($weightsInput, $inputVector);
        $hiddenWeighted = $this->matrixMultiply($weightsHidden, $prevHiddenState);

        $weightedSum = $this->matrixAddition($inputWeighted, $hiddenWeighted);
        $weightedSumWithBias = $this->matrixAddition($weightedSum, $bias);

        // Apply activation function (e.g., Tanh)
        $hiddenState = $this->applyActivation($weightedSumWithBias, $activation);

        return $hiddenState;
    }

    /**
     * recommendItems
     * Uses an artificial neural network to recommend items based on user data.
	 * recommendItems: This method is a placeholder for implementing recommendation logic using ANNs. It receives user data as input and aims to generate recommendations based on that data.
	 * $userData: Represents historical user data or user profiles, including preferences, behaviors, past interactions, etc. This data is used to train the neural network or to make predictions for generating recommendations.
	 * $recommendedItems: Placeholder variable to store the recommended items generated by the recommendation system.
     */
    public function recommendItems($userData) {
        // Implement the recommendation logic using ANNs
        // Use user historical data ($userData) to train or make predictions

        // Example: Placeholder logic
        $recommendedItems = []; // Placeholder for recommended items
        
        // Apply ANN logic to generate recommendations based on user data

        return $recommendedItems;
    }	
	
/////////////////////////////////////NLP logic using ANNs/////////////////////////////////////

    /**
     * performNLP
     * Utilizes ANNs for Natural Language Processing tasks.
	 * performNLP: The method acts as a generic NLP processor using ANNs, with a switch-case structure to handle different NLP tasks.
	 * performSentimentAnalysis: Placeholder method to perform sentiment analysis using ANNs. You'd replace the logic inside this method with a trained sentiment analysis model.
	 * performTranslation: Placeholder method to perform language translation using ANNs. Similarly, you'd replace this with a trained translation model.
	 * Developers using PHPNeuroForge can call the performNLP method with text data and a specified task, and based on the provided task, the utility will invoke the corresponding NLP logic using ANNs. Adjust the placeholder logic with actual trained models for accurate NLP processing.
////////////////////////////////////////// Example Of Use //////////////////////////////////////////
// Import the Utilities class from utilities.php
require_once 'utilities.php';

// Example usage of performNLP for sentiment analysis and translation
$text = "This is a great product! I love it.";

// Create an instance of the Utilities class
$util = new Utilities();

// Perform sentiment analysis
$sentiment = $util->performNLP($text, 'sentiment_analysis');
echo "Sentiment: $sentiment<br>";

// Perform language translation
$translatedText = $util->performNLP($text, 'translation');
echo "Translated Text: $translatedText<br>";
////////////////////////////////////////// End Example Of Use //////////////////////////////////////////

     */
    public function performNLP($textData, $task) {
        // Implement the NLP logic using ANNs
        // $textData: Input text for NLP processing
        // $task: Specific NLP task to perform (sentiment analysis, translation, etc.)

        // Placeholder logic for different NLP tasks using ANNs
        switch ($task) {
            case 'sentiment_analysis':
                // Placeholder for sentiment analysis using ANNs
                $processedData = $this->performSentimentAnalysis($textData);
                break;
            case 'translation':
                // Placeholder for language translation using ANNs
                $processedData = $this->performTranslation($textData);
                break;
            // Add more cases for other NLP tasks as needed
            default:
                $processedData = null; // No valid task provided
                break;
        }

        return $processedData;
    }

    /**
     * performSentimentAnalysis
     * Placeholder method for sentiment analysis using ANNs.
     */
    private function performSentimentAnalysis($textData) {
        // Perform sentiment analysis using ANNs (placeholder logic)
        // $textData: Input text for sentiment analysis

        // Perform analysis and return the sentiment
        $sentiment = 'Positive'; // Placeholder sentiment result

        return $sentiment;
    }

    /**
     * performTranslation
     * Placeholder method for language translation using ANNs.
     */
    private function performTranslation($textData) {
        // Perform language translation using ANNs (placeholder logic)
        // $textData: Input text for translation

        // Perform translation and return the translated text
        $translatedText = 'Translated text'; // Placeholder translation result

        return $translatedText;
    }	

/////////////////////////////////////END NLP logic using ANNs/////////////////////////////////////
	
/////////////////////////////////////Image and Video Processing/////////////////////////////////////

    /**
     * Perform image recognition using ANNs
     * 
     * @param string $imagePath Path to the image file
     * @return string Recognized objects or content in the image
	 * Enhanced methods: The performImageRecognition and analyzeVideoContent methods now call placeholder private methods (callImageRecognitionAPI and callVideoAnalysisAPI) to simulate the process of calling external APIs for image recognition and video analysis. Developers can replace these methods with actual API calls or integrate with their preferred image recognition and video analysis libraries.
	 * Private helper methods: The callImageRecognitionAPI and callVideoAnalysisAPI methods serve as placeholders for calling external APIs. Developers should replace these methods with their specific implementation or library integration.
////////////////////////////////////////// Example Of Use //////////////////////////////////////////
// Include or require the Utilities class file
require_once('Utilities.php');

// Create an instance of the Utilities class
$utilities = new Utilities();

// Example: Perform image recognition
$imagePath = 'path/to/your/image.jpg';
$imageRecognitionResult = $utilities->performImageRecognition($imagePath);
echo "Image Recognition Result: $imageRecognitionResult <br>";

// Example: Analyze video content
$videoPath = 'path/to/your/video.mp4';
$videoAnalysisResult = $utilities->analyzeVideoContent($videoPath);
echo "Video Analysis Result: $videoAnalysisResult";
////////////////////////////////////////// END Example Of Use //////////////////////////////////////////

     */
    public function performImageRecognition($imagePath) {
        // Placeholder logic for image recognition using ANNs
        // Developers should replace this with actual image recognition implementation
        $recognizedObjects = $this->callImageRecognitionAPI($imagePath);

        return "Detected objects: " . implode(', ', $recognizedObjects);
    }

    /**
     * Analyze video content using ANNs
     * 
     * @param string $videoPath Path to the video file
     * @return string Analysis or detected elements in the video
     */
    public function analyzeVideoContent($videoPath) {
        // Placeholder logic for video content analysis using ANNs
        // Developers should replace this with actual video analysis implementation
        $videoAnalysisResults = $this->callVideoAnalysisAPI($videoPath);

        return "Video content analysis: " . implode(', ', $videoAnalysisResults);
    }

    /**
     * Placeholder method for calling an image recognition API
     * 
     * @param string $imagePath Path to the image file
     * @return array Recognized objects
     */
    private function callImageRecognitionAPI($imagePath) {
        // Placeholder logic to call an image recognition API
        // Developers should replace this with their API integration
        // Example: return ['cat', 'car', 'tree'];
        return ['object1', 'object2', 'object3'];
    }

    /**
     * Placeholder method for calling a video analysis API
     * 
     * @param string $videoPath Path to the video file
     * @return array Analysis results
     */
    private function callVideoAnalysisAPI($videoPath) {
        // Placeholder logic to call a video analysis API
        // Developers should replace this with their API integration
        // Example: return ['action scenes', 'beach view'];
        return ['result1', 'result2'];
    }
}

/////////////////////////////////////END Image and Video Processing/////////////////////////////////////

}
?>
