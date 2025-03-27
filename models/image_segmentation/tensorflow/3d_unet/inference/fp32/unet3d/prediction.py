import os

import nibabel as nib
import numpy as np
import tables
import time
import math

from .training import load_old_model
from .utils import pickle_load
from .utils.patches import (
    reconstruct_from_patches,
    get_patch_from_3d_data,
    compute_patch_indices,
)
from .augment import permute_data, generate_permutation_keys, reverse_permute_data


def patch_wise_prediction(model, data, overlap=0, batch_size=1, permute=False):
    """
    :param batch_size:
    :param model:
    :param data:
    :param overlap:
    :return:
    """
    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    predictions = list()
    indices = compute_patch_indices(
        data.shape[-3:], patch_size=patch_shape, overlap=overlap
    )
    batch = list()
    i = 0

    total_model_time = 0
    while i < len(indices):
        while len(batch) < batch_size:
            patch = get_patch_from_3d_data(
                data[0], patch_shape=patch_shape, patch_index=indices[i]
            )
            batch.append(patch)
            i += 1

        # print('batch.shape: {}'.format(np.asarray(batch).shape))
        start_time = time.time()
        prediction = predict(model, np.asarray(batch), permute=permute)
        end_time = time.time()
        total_model_time += end_time - start_time

        batch = list()
        # print('prediction.shape: {}'.format(prediction.shape))
        for predicted_patch in prediction:
            # print('predicted_patch.shape: {}'.format(predicted_patch.shape))
            predictions.append(predicted_patch)

    # print('model evaluation time: {} ms'.format(total_model_time * 1000))
    # print('predictions.length: {}'.format(len(predictions)))
    # print('predictions[0].shape: {}'.format(predictions[0].shape))
    output_shape = [int(model.output.shape[1])] + list(data.shape[-3:])
    return reconstruct_from_patches(
        predictions, patch_indices=indices, data_shape=output_shape
    )


def get_prediction_labels(prediction, threshold=0.5, labels=None):
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def get_test_indices(testing_file):
    return pickle_load(testing_file)


def predict_from_data_file(model, open_data_file, index):
    return model.predict(open_data_file.root.data[index])


def predict_and_get_image(model, data, affine):
    return nib.Nifti1Image(model.predict(data)[0, 0], affine)


def predict_from_data_file_and_get_image(model, open_data_file, index):
    return predict_and_get_image(
        model, open_data_file.root.data[index], open_data_file.root.affine
    )


def predict_from_data_file_and_write_image(model, open_data_file, index, out_file):
    image = predict_from_data_file_and_get_image(model, open_data_file, index)
    image.to_filename(out_file)


def prediction_to_image(
    prediction, affine, label_map=False, threshold=0.5, labels=None
):
    if prediction.shape[1] == 1:
        data = prediction[0, 0]
        if label_map:
            label_map_data = np.zeros(prediction[0, 0].shape, np.int8)
            if labels:
                label = labels[0]
            else:
                label = 1
            label_map_data[data > threshold] = label
            data = label_map_data
    elif prediction.shape[1] > 1:
        if label_map:
            label_map_data = get_prediction_labels(
                prediction, threshold=threshold, labels=labels
            )
            data = label_map_data[0]
        else:
            return multi_class_prediction(prediction, affine)
    else:
        raise RuntimeError(
            "Invalid prediction array shape: {0}".format(prediction.shape)
        )
    return nib.Nifti1Image(data, affine)


def multi_class_prediction(prediction, affine):
    prediction_images = []
    for i in range(prediction.shape[1]):
        prediction_images.append(nib.Nifti1Image(prediction[0, i], affine))
    return prediction_images


def run_validation_case(
    data_index,
    output_dir,
    model,
    data_file,
    training_modalities,
    output_label_map=False,
    threshold=0.5,
    labels=None,
    overlap=16,
    permute=False,
):
    """
    Runs a test case and writes predicted images to file.
    :param data_index: Index from of the list of test cases to get an image prediction from.
    :param output_dir: Where to write prediction images.
    :param output_label_map: If True, will write out a single image with one or more labels. Otherwise outputs
    the (sigmoid) prediction values from the model.
    :param threshold: If output_label_map is set to True, this threshold defines the value above which is
    considered a positive result and will be assigned a label.
    :param labels:
    :param training_modalities:
    :param data_file:
    :param model:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine = data_file.root.affine[data_index]
    test_data = np.asarray([data_file.root.data[data_index]])
    # print('test_data.shape: {}'.format(test_data.shape))
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    test_truth = nib.Nifti1Image(data_file.root.truth[data_index][0], affine)
    test_truth.to_filename(os.path.join(output_dir, "truth.nii.gz"))

    patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
    if patch_shape == test_data.shape[-3:]:
        # print('this branch !!!!!!!!!!!!!')
        prediction = predict(model, test_data, permute=permute)
    else:
        prediction = patch_wise_prediction(
            model=model, data=test_data, overlap=overlap, permute=permute
        )[np.newaxis]
    # print('!!!!!prediction.shape: {}'.format(prediction.shape))
    prediction_image = prediction_to_image(
        prediction,
        affine,
        label_map=output_label_map,
        threshold=threshold,
        labels=labels,
    )
    if isinstance(prediction_image, list):
        for i, image in enumerate(prediction_image):
            image.to_filename(
                os.path.join(output_dir, "prediction_{0}.nii.gz".format(i + 1))
            )
    else:
        prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))


def run_validation_cases(
    validation_keys_file,
    model_file,
    training_modalities,
    labels,
    hdf5_file,
    output_label_map=False,
    output_dir=".",
    threshold=0.5,
    overlap=16,
    permute=False,
    warmup=10,
    report_interval=1,
    batch_size=1,
    n_batch=10,
):
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    data_file = tables.open_file(hdf5_file, "r")

    elapsed_time = 0
    elapsed_step = 0

    for index in validation_indices:
        start = time.time()
        if "subject_ids" in data_file.root:
            case_directory = os.path.join(
                output_dir, data_file.root.subject_ids[index].decode("utf-8")
            )
        else:
            case_directory = os.path.join(
                output_dir, "validation_case_{}".format(index)
            )
        run_validation_case(
            data_index=index,
            output_dir=case_directory,
            model=model,
            data_file=data_file,
            training_modalities=training_modalities,
            output_label_map=output_label_map,
            labels=labels,
            threshold=threshold,
            overlap=overlap,
            permute=permute,
        )
        end = time.time()

        if index >= warmup:
            elapsed_time += end - start
            elapsed_step += 1

            if elapsed_step + warmup == n_batch:
                # print('performance = {} img/s, count for {} steps and batch size is {}'.format(elapsed_step * batch_size / elapsed_time, elapsed_step, batch_size), flush=True)
                # print('latency = {} ms'.format(1000 * elapsed_time / elapsed_step), flush=True)
                print(
                    "Time spent per BATCH: %.4f ms"
                    % (1000.0 * elapsed_time / elapsed_step)
                )
                print(
                    "Total samples/sec: %.4f samples/s"
                    % (elapsed_step * batch_size / elapsed_time)
                )
                break

    data_file.close()


def predict(model, data, permute=False):
    if permute:
        predictions = list()
        for batch_index in range(data.shape[0]):
            predictions.append(predict_with_permutations(model, data[batch_index]))
        return np.asarray(predictions)
    else:
        return model.predict(data)


def predict_with_permutations(model, data):
    predictions = list()
    for permutation_key in generate_permutation_keys():
        temp_data = permute_data(data, permutation_key)[np.newaxis]
        predictions.append(
            reverse_permute_data(model.predict(temp_data)[0], permutation_key)
        )
    return np.mean(predictions, axis=0)


def run_large_batch_validation_cases(
    validation_keys_file,
    model_file,
    training_modalities,
    labels,
    hdf5_file,
    output_label_map=False,
    output_dir=".",
    threshold=0.5,
    overlap=16,
    permute=False,
    batch_size=1,
    warmup=1,
    report_interval=1,
    n_batch=10,
):
    validation_indices = pickle_load(validation_keys_file)
    model = load_old_model(model_file)
    data_file = tables.open_file(hdf5_file, "r")

    #
    # Initilize validation case directory:
    #
    # for index in validation_indices:
    #     if 'subject_ids' in data_file.root:
    #         case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
    #     else:
    #         case_directory = os.path.join(output_dir, "validation_case_{}".format(index))

    #     if not os.path.exists(case_directory):
    #         os.makedirs(case_directory)

    #     # Write image to validation case directory:
    #     affine = data_file.root.affine[index]
    #     affine_dict[index] = affine
    #     test_data = np.asarray([data_file.root.data[index]])
    #     for i, modality in enumerate(training_modalities):
    #         image = nib.Nifti1Image(test_data[0, i], affine)
    #         image.to_filename(os.path.join(case_directory, "data_{0}.nii.gz".format(modality)))

    #     test_truth = nib.Nifti1Image(data_file.root.truth[index][0], affine)
    #     test_truth.to_filename(os.path.join(case_directory, "truth.nii.gz"))

    step = math.ceil(len(validation_indices) / batch_size)

    elapsed_time = 0
    elapsed_step = 0

    for i in range(step):
        print("iteration {} ...".format(i))

        start_time = time.time()

        test_data_index = validation_indices[i * batch_size : (i + 1) * batch_size]
        test_data = []

        affine_dict = {}

        for tdi in test_data_index:
            #
            # Initilize validation case directory:
            #
            if "subject_ids" in data_file.root:
                case_directory = os.path.join(
                    output_dir, data_file.root.subject_ids[tdi].decode("utf-8")
                )
            else:
                case_directory = os.path.join(
                    output_dir, "validation_case_{}".format(tdi)
                )

            if not os.path.exists(case_directory):
                os.makedirs(case_directory)

            # Write image to validation case directory:
            affine = data_file.root.affine[tdi]
            affine_dict[tdi] = affine
            test_data_elem = np.asarray([data_file.root.data[tdi]])
            for index, modality in enumerate(training_modalities):
                image = nib.Nifti1Image(test_data_elem[0, index], affine)
                image.to_filename(
                    os.path.join(case_directory, "data_{0}.nii.gz".format(modality))
                )

            test_truth = nib.Nifti1Image(data_file.root.truth[tdi][0], affine)
            test_truth.to_filename(os.path.join(case_directory, "truth.nii.gz"))

            test_data.append(data_file.root.data[tdi])

        test_data = np.asarray([test_data])
        # print('test_data.shape: {}'.format(test_data.shape))

        patch_shape = tuple([int(dim) for dim in model.input.shape[-3:]])
        if patch_shape == test_data.shape[-3:]:
            # prediction = predict(model, test_data, permute=permute)
            if test_data.ndim is 6:
                assert test_data.shape[0] is 1
                test_data = test_data[0]
            predictions = predict(model, test_data, permute=permute)
        else:
            predictions = []
            indices = compute_patch_indices(
                test_data.shape[-3:], patch_size=patch_shape, overlap=overlap
            )
            batch = []

            # print('len(indices): {}'.format(len(indices)))

            for b in range(test_data.shape[1]):
                indices_index = 0
                while indices_index < len(indices):
                    patch = get_patch_from_3d_data(
                        test_data[0][b],
                        patch_shape=patch_shape,
                        patch_index=indices[indices_index],
                    )
                    batch.append(patch)
                    indices_index += 1

            pred_start = time.time()
            prediction = predict(model, np.asarray(batch), permute=permute)
            pred_stop = time.time()
            print("pred time: {} ms".format((pred_stop - pred_start) * 1000))
            # print('prediction.shape: {}'.format(prediction.shape))
            # batch = []
            ps = prediction.shape
            assert ps[0] % test_data.shape[1] == 0
            prediction = np.reshape(
                prediction,
                (
                    test_data.shape[1],
                    int(ps[0] / test_data.shape[1]),
                    ps[1],
                    ps[2],
                    ps[3],
                    ps[4],
                ),
            )

            for batch_index, batch_prediction in enumerate(prediction):
                # in case of the list out of index situation
                if len(predictions) < (batch_index + 1):
                    assert batch_index is len(predictions)
                    predictions.append([])

                for patch_index, predicted_patch in enumerate(batch_prediction):
                    predictions[batch_index].append(predicted_patch)

            output_shape = [int(model.output.shape[1])] + list(test_data.shape[-3:])

            #
            # Re-construction
            #
            reconstructed_predictions = []
            for pred in predictions:
                # print('before reconstruction: {}, {}'.format(pred[0].shape, len(pred)))
                reconstructed_prediction = reconstruct_from_patches(
                    pred, patch_indices=indices, data_shape=output_shape
                )[np.newaxis]
                # print('reconstructed_prediction.shape: {}'.format(reconstructed_prediction.shape))
                reconstructed_predictions.append(reconstructed_prediction)

            #
            # Predict to image
            #
            prediction_images = []
            for pred_index, pred in enumerate(reconstructed_predictions):
                rec_pred_index = test_data_index[pred_index]
                # print('pred_index: {}'.format(rec_pred_index))

                affine = affine_dict[rec_pred_index]

                # print('pred.shape: {}'.format(pred.shape))
                prediction_image = prediction_to_image(
                    pred,
                    affine,
                    label_map=output_label_map,
                    threshold=threshold,
                    labels=labels,
                )

                prediction_images.append(prediction_images)

                if "subject_ids" in data_file.root:
                    case_directory = os.path.join(
                        output_dir,
                        data_file.root.subject_ids[rec_pred_index].decode("utf-8"),
                    )
                else:
                    case_directory = os.path.join(
                        output_dir, "validation_case_{}".format(rec_pred_index)
                    )
                if isinstance(prediction_image, list):
                    for image_index, image in enumerate(prediction_image):
                        image.to_filename(
                            os.path.join(
                                case_directory,
                                "prediction_{0}.nii.gz".format(image_index + 1),
                            )
                        )
                else:
                    prediction_image.to_filename(
                        os.path.join(case_directory, "prediction.nii.gz")
                    )

        stop_time = time.time()

        if i >= warmup:
            elapsed_time += stop_time - start_time
            elapsed_step += 1

            if elapsed_step + warmup == n_batch:
                print(
                    "performance = {} img/s, count for {} steps and batch size is {}".format(
                        elapsed_step * batch_size / elapsed_time,
                        elapsed_step,
                        batch_size,
                    )
                )
                print("latency = {} ms".format(1000 * elapsed_time / elapsed_step))
                elapsed_time = 0
                elapsed_step = 0
                break

    data_file.close()
