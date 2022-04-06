import numpy as np


def tensor_to_numpy(j):
    array = np.empty(j["size"])
    array[:] = j["vals"]
    print(tuple(j['dims']))
    return np.reshape(array, tuple(j["dims"]))


def read_frequency_proto_iter_data(my_iter_data, num_states, num_orbitals):
    num_iters = len(my_iter_data)
    dres = np.empty((num_iters, num_states))
    res_X = np.empty((num_iters, num_states))
    res_Y = np.empty((num_iters, num_states))
    polar = np.empty((num_iters, 3, 3))
    for i in range(num_iters):
        dres[i, :] = tensor_to_numpy(my_iter_data[i]["density_residuals"])
        res_X[i, :] = tensor_to_numpy(my_iter_data[i]["res_X"])
        res_Y[i, :] = tensor_to_numpy(my_iter_data[i]["res_Y"])
        polar[i, :, :] = tensor_to_numpy(my_iter_data[i]["polar"])
    data = {}
    names = ["density_residuals", "res_X", "res_Y", "polar"]
    vals = [dres, res_X, res_Y, polar]
    for name, val in zip(names, vals):
        data[name] = val
    return data


def read_excited_proto_iter_data(my_iter_data, num_states, num_orbitals):
    num_iters = len(my_iter_data)
    dres = np.empty((num_iters, num_states))
    res_X = np.empty((num_iters, num_states))
    res_Y = np.empty((num_iters, num_states))
    omega = np.empty((num_iters, num_states))
    for i in range(num_iters):
        dres[i, :] = tensor_to_numpy(my_iter_data[i]["density_residuals"])
        res_X[i, :] = tensor_to_numpy(my_iter_data[i]["res_X"])
        res_Y[i, :] = tensor_to_numpy(my_iter_data[i]["res_Y"])
        omega[i, :] = tensor_to_numpy(my_iter_data[i]["omega"])
    data = {}
    names = ["density_residuals", "res_X", "res_Y", "omega"]
    vals = [dres, res_X, res_Y, omega]
    for name, val in zip(names, vals):
        data[name] = val
    return data


# input response_info json and returns a dict of response paramters
# and a list of dicts of numpy arrays holding response data
    protocol_data = response_info['protocol_data']
    response_parameters = response_info['response_parameters']
    n_states = response_parameters["states"]
    n_orbitals = response_parameters["num_orbitals"]
    num_protos = len(protocol_data)
    protos = []
    proto_data = []
    for p in range(num_protos):
        protos.append(protocol_data[p]["proto"])
        iter_data = protocol_data[p]["iter_data"]
        if response_parameters["excited_state"]:
            proto_data.append(read_excited_proto_iter_data(
                iter_data, n_states, n_orbitals))
        else:
            proto_data.append(read_frequency_proto_iter_data(
                iter_data, n_states, n_orbitals))
    return response_parameters, proto_data
