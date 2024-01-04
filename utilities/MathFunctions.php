<?php
// utilities/MathFunctions.php
// Auther: Majdi M. S. Awad
// Year: 2024
// Version: 1.0.0
// MIT License

/**
 * Matrix Multiplication Function
 * Performs matrix multiplication.
 *
 * @param array $matrix1    First matrix.
 * @param array $matrix2    Second matrix.
 * @return array            Resultant matrix after multiplication.
 */
function matrixMultiply($matrix1, $matrix2) {
    // Ensure matrix dimensions are compatible for multiplication
    $rowsA = count($matrix1);
    $colsA = count($matrix1[0]);
    $rowsB = count($matrix2);
    $colsB = count($matrix2[0]);

    if ($colsA !== $rowsB) {
        // Throw an error if matrix dimensions are incompatible
        throw new Exception("Matrix dimensions are incompatible for multiplication.");
    }

    // Perform matrix multiplication
    $result = [];
    for ($i = 0; $i < $rowsA; $i++) {
        for ($j = 0; $j < $colsB; $j++) {
            $result[$i][$j] = 0;
            for ($k = 0; $k < $colsA; $k++) {
                $result[$i][$j] += $matrix1[$i][$k] * $matrix2[$k][$j];
            }
        }
    }
    return $result;
}

/**
 * Elementwise Operation Function
 * Performs elementwise operations (addition, subtraction, etc.) on matrices or vectors.
 *
 * @param array $matrix1    First matrix or vector.
 * @param array $matrix2    Second matrix or vector.
 * @param string $operation Operation to perform ('add', 'subtract', 'multiply', etc.).
 * @return array            Resultant matrix or vector after the operation.
 */
function elementwiseOperation($matrix1, $matrix2, $operation) {
    $result = [];

    // Check if the matrices have compatible dimensions for elementwise operations
    $rowsM1 = count($matrix1);
    $colsM1 = count($matrix1[0]);
    $rowsM2 = count($matrix2);
    $colsM2 = count($matrix2[0]);

    if ($rowsM1 !== $rowsM2 || $colsM1 !== $colsM2) {
        // Throw an error if dimensions are incompatible
        throw new Exception("Matrix dimensions are incompatible for elementwise operation.");
    }

    // Perform elementwise operation based on the specified operation type
    for ($i = 0; $i < $rowsM1; $i++) {
        for ($j = 0; $j < $colsM1; $j++) {
            switch ($operation) {
                case 'add':
                    $result[$i][$j] = $matrix1[$i][$j] + $matrix2[$i][$j];
                    break;
                case 'subtract':
                    $result[$i][$j] = $matrix1[$i][$j] - $matrix2[$i][$j];
                    break;
                case 'multiply':
                    $result[$i][$j] = $matrix1[$i][$j] * $matrix2[$i][$j];
                    break;
                // Add more operations (division, exponentiation, etc.) as needed
                default:
                    throw new Exception("Unsupported elementwise operation.");
            }
        }
    }
    return $result;
}

/**
 * Activation Derivative Function
 * Computes the derivative of an activation function.
 *
 * @param string $activation    Name of the activation function ('sigmoid', 'relu', 'tanh', etc.).
 * @param array $input          Input values.
 * @return array                Derivative values.
 */
function activationDerivative($activation, $input) {
    $derivative = [];

    // Calculate derivative based on the specified activation function
    switch ($activation) {
        case 'sigmoid':
            // Example derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x))
            foreach ($input as $val) {
                $derivative[] = $val * (1 - $val);
            }
            break;
        case 'relu':
            // Example derivative of the ReLU function: f'(x) = 1 if x > 0, else 0
            foreach ($input as $val) {
                $derivative[] = ($val > 0) ? 1 : 0;
            }
            break;
        case 'tanh':
            // Example derivative of the tanh function: f'(x) = 1 - f(x)^2
            foreach ($input as $val) {
                $derivative[] = 1 - ($val * $val);
            }
            break;
        // Add derivatives for more activation functions if needed
        default:
            throw new Exception("Derivative calculation not supported for the specified activation function.");
    }
    return $derivative;
}
?>
