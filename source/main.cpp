#include <stdio.h>

#include "edge-impulse-sdk/classifier/ei_run_classifier.h"

// Callback function declaration
static int get_signal_data(size_t offset, size_t length, float *out_ptr);

// Raw features copied from test sample
static const float features[] = {
    // Copy raw features here (e.g. from the 'Model testing' page)
    3.0400, 0.6400, -2.6400, 2.5000, 1.6800, -0.6300, 2.5000, 1.6800, -0.6300, 2.8700, 1.9100, 1.5000, 2.7600, 0.9200, 3.2100, 2.2700, 0.2400, 5.3400, 1.9000, -0.0200, 6.9200, 2.1900, 0.9800, 6.9900, 2.8900, 2.1500, 7.5800, 2.8900, 2.1500, 7.5800, 2.2500, 1.6300, 9.6200, 2.2300, 1.1400, 13.1900, 3.2900, 1.6300, 16.2400, 3.9100, 2.1400, 13.5000, 3.9800, 3.8500, 10.5300, 3.2600, 2.7700, 14.7100, 3.2600, 2.7700, 14.7100, 2.5600, 1.5900, 14.9100, 2.1800, 0.8300, 13.5500, 3.5800, 2.0800, 14.9600, 3.9500, 2.0900, 17.2600, 3.2400, 0.7600, 17.6000, 2.3300, 0.4700, 16.3200, 2.3300, 0.4700, 16.3200, 1.8600, 0.6500, 13.6600, 2.1000, 0.9200, 13.4500, 2.0400, -0.5800, 16.3400, 1.6300, -1.5500, 18.3900, 1.7600, -1.1700, 19.9700, 1.1700, -1.6800, 19.9800, 1.1700, -1.6800, 19.9800, 0.9000, -2.5700, 19.9800, 1.1300, -2.2600, 19.9800, 1.0100, -2.0700, 19.9800, 1.1700, -1.2600, 19.9800, 1.3900, -0.8300, 19.9800, 1.4300, -0.7200, 19.9800, 1.4300, -0.7200, 19.9800, 1.0200, -0.1400, 19.9700, 0.4600, 0.5600, 18.0200, 0.4100, 1.9000, 15.6500, -0.0100, 2.4600, 13.6200, 0.1100, 2.6800, 11.4200, 0.3500, 2.5500, 9.7700, 0.3500, 2.5500, 9.7700, 0.2400, 2.4000, 7.5600, 0.5700, 1.9400, 4.6200, 0.6900, 0.8400, 2.5400, 0.3600, -0.1700, 2.1100, 1.0500, 0.3800, 2.1000, 2.1900, 1.2900, 2.4000, 2.1900, 1.2900, 2.4000, 2.4300, 0.1400, 2.6000, 2.0900, -0.9900, 1.9600, 2.3100, -1.3800, 0.2500, 2.8500, -0.5500, -1.2200, 3.2900, -0.3500, -2.1600, 3.3900, -0.5200, -2.0800, 3.3900, -0.5200, -2.0800, 4.1900, -0.2100, -1.4700, 4.5000, -0.1600, -1.9700, 4.4400, 0.1100, -3.1700, 4.4000, 0.6300, -4.0200, 4.7100, 0.7900, -4.1400, 4.8300, 0.8100, -4.4000, 4.8300, 0.8100, -4.4000, 4.9400, 1.3400, -4.2900, 4.8100, 2.2400, -2.6800, 4.6600, 2.4500, -0.6400, 4.4900, 2.6400, 1.9000, 3.4900, 1.2400, 2.4100, 3.4900, 1.2400, 2.4100, 3.1800, 0.6600, 2.3700, 4.1000, 2.4200, 4.5900, 4.6900, 2.6200, 8.1700, 3.1200, 0.7700, 8.0900, 3.3500, 0.6800, 7.3200, 3.6700, 1.4300, 8.8100, 3.6700, 1.4300, 8.8100, 3.3500, 0.9700, 11.0400, 2.8500, 0.2200, 12.0900, 2.4300, 0.4700, 11.8000, 2.6000, 1.0700, 12.0700, 1.9800, 1.0000, 11.5700, 1.4100, 0.3900, 12.9100, 1.4100, 0.3900, 12.9100, 2.3900, 0.4200, 14.6800, 2.0900, -0.1600, 14.0800, 2.0000, -0.1000, 13.8900, 1.5600, -0.1400, 12.7000, 1.6000, 0.5800, 12.8800, 1.5300, 1.4400, 13.7000, 1.5300, 1.4400, 13.7000, 1.0900, 2.2700, 12.7400, 0.5300, 2.6700, 11.5200, 0.6900, 2.3000, 13.1300, 0.3600, 0.9800, 16.2400, -0.0800, -0.4000, 19.1900, -0.1400, -1.1000, 19.9800, -0.1400, -1.1000, 19.9800, -0.0500, -1.9400, 19.9800, -0.2500, -2.7100, 19.9800, -0.1400, -3.0800, 19.9800, -0.0700, -2.4800, 19.9800, -0.2200, -1.7800, 19.9800, -0.1900, -1.5000, 19.9800, -0.1900, -1.5000, 19.9800, -0.1000, -1.4300, 19.9800, -0.2700, -1.0600, 19.8000, 0.1200, 0.5700, 18.0400, -0.1100, 1.5100, 15.6900, -0.6400, 1.4400, 13.6800, 0.1300, 1.8700, 13.8700, 0.1300, 1.8700, 13.8700, -0.1900, 1.3900, 12.7900, 0.1200, 1.5700, 10.2800, 0.1300, 1.3600, 7.5500, 0.3500, 0.3400, 6.4900, 0.1500, -0.2500, 6.6500, 0.0400, -0.2600, 4.9500, 0.0400, -0.2600, 4.9500, -0.1400, -0.3800, 2.5600, 0.4300, 0.7100, 1.1200, 1.2600, 1.5500, -0.1700, 1.6300, 0.7400, -2.1200
};

int main(int argc, char **argv) {

    signal_t signal;            // Wrapper for raw input buffer
    ei_impulse_result_t result; // Used to store inference output
    EI_IMPULSE_ERROR res;       // Return code from inference

    // Calculate the length of the buffer
    size_t buf_len = sizeof(features) / sizeof(features[0]);

    // Make sure that the length of the buffer matches expected input length
    if (buf_len != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        ei_printf("ERROR: The size of the input buffer is not correct.\r\n");
        ei_printf("Expected %d items, but got %d\r\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE,
                (int)buf_len);
        return 1;
    }

    // Assign callback function to fill buffer used for preprocessing/inference
    signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
    signal.get_data = &get_signal_data;

    // Perform DSP pre-processing and inference
    res = run_classifier(&signal, &result, false);

    // Print return code and how long it took to perform inference
    ei_printf("run_classifier returned: %d\r\n", res);
    ei_printf("Timing: DSP %d ms, inference %d ms, anomaly %d ms\r\n",
            result.timing.dsp,
            result.timing.classification,
            result.timing.anomaly);

    // Print the prediction results (object detection)
#if EI_CLASSIFIER_OBJECT_DETECTION == 1
    ei_printf("Object detection bounding boxes:\r\n");
    for (uint32_t i = 0; i < result.bounding_boxes_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.bounding_boxes[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }

    // Print the prediction results (classification)
#else
    ei_printf("Predictions:\r\n");
    for (uint16_t i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        ei_printf("  %s: ", ei_classifier_inferencing_categories[i]);
        ei_printf("%.5f\r\n", result.classification[i].value);
    }
#endif

    // Print anomaly result (if it exists)
#if EI_CLASSIFIER_HAS_ANOMALY
    ei_printf("Anomaly prediction: %.3f\r\n", result.anomaly);
#endif

#if EI_CLASSIFIER_HAS_VISUAL_ANOMALY
    ei_printf("Visual anomalies:\r\n");
    for (uint32_t i = 0; i < result.visual_ad_count; i++) {
        ei_impulse_result_bounding_box_t bb = result.visual_ad_grid_cells[i];
        if (bb.value == 0) {
            continue;
        }
        ei_printf("  %s (%f) [ x: %u, y: %u, width: %u, height: %u ]\r\n",
                bb.label,
                bb.value,
                bb.x,
                bb.y,
                bb.width,
                bb.height);
    }
    ei_printf("Visual anomaly values: Mean : %.3f Max : %.3f\r\n",
    result.visual_ad_result.mean_value, result.visual_ad_result.max_value);
#endif

    return 0;
}

// Callback: fill a section of the out_ptr buffer when requested
static int get_signal_data(size_t offset, size_t length, float *out_ptr) {
    for (size_t i = 0; i < length; i++) {
        out_ptr[i] = (features + offset)[i];
    }

    return EIDSP_OK;
}
