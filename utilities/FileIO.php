<?php
// utilities/FileIO.php
// Auther: Majdi M. S. Awad
// Year: 2024
// Version: 1.0.0
// MIT License

/**
 * ModelIO Class
 * Handles reading and writing of neural network models.
 */
class ModelIO {
    /**
     * saveModel
     * Saves the neural network model to a file.
     *
     * @param object $model     Neural network model object to be saved.
     * @param string $fileName  Name of the file to save the model.
     * @return bool             Returns true on successful model save, false otherwise.
     */
    public static function saveModel($model, $fileName) {
        // Serialize the model object and save it to a file
        $serializedModel = serialize($model);
        return file_put_contents($fileName, $serializedModel) !== false;
    }

    /**
     * loadModel
     * Loads a neural network model from a file.
     *
     * @param string $fileName  Name of the file containing the saved model.
     * @return object|null      Returns the loaded model object or null if file loading fails.
     */
    public static function loadModel($fileName) {
        // Check if the file exists
        if (file_exists($fileName)) {
            // Read the serialized model data from the file and unserialize it
            $serializedModel = file_get_contents($fileName);
            return unserialize($serializedModel);
        } else {
            return null; // File doesn't exist or cannot be loaded
        }
    }
}

/**
 * DataIO Class
 * Handles reading and writing of data for neural network operations.
 */
class DataIO {
    /**
     * readCSV
     * Reads data from a CSV file and returns it as an array.
     *
     * @param string $fileName  Name of the CSV file.
     * @return array|bool       Returns the data array if successful, false otherwise.
     */
    public static function readCSV($fileName) {
        $data = [];

        // Check if the file exists and is readable
        if (file_exists($fileName) && is_readable($fileName)) {
            // Read data from the CSV file
            $fileHandle = fopen($fileName, 'r');
            while (($row = fgetcsv($fileHandle)) !== false) {
                $data[] = $row;
            }
            fclose($fileHandle);
            return $data;
        } else {
            return false; // File doesn't exist or cannot be read
        }
    }

    /**
     * writeCSV
     * Writes data to a CSV file.
     *
     * @param array $data       Data array to be written to the CSV file.
     * @param string $fileName  Name of the CSV file to write the data.
     * @return bool             Returns true on successful write, false otherwise.
     */
    public static function writeCSV($data, $fileName) {
        // Open the file for writing
        $fileHandle = fopen($fileName, 'w');

        // Write data to the CSV file
        foreach ($data as $row) {
            fputcsv($fileHandle, $row);
        }

        // Close the file handle
        fclose($fileHandle);

        return true;
    }
}
?>
