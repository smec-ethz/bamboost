import numpy as np
import argparse

from dbmanager.simulation import SimulationWriter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--uid", type=str)
    args = parser.parse_args()
    uid, path = args.uid, args.path

    # Use with statement for keeping track of the status of the simulation
    with SimulationWriter(uid, path) as writer:
        writer.register_git_attributes()
        writer.add_metadata()

        # to load the parameters of this simulation use
        params = writer.parameters

        # do what you have to do
        # here we create a simple mesh
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 2.0]])

        initial_values = np.random.randint(0, 10, 5)

        for t in range(params["timesteps"]):
            result = initial_values * t ** params["exponent"]

            # write the coordinates
            # if you have a mesh with coords and connectivity, you can add a global mesh with
            # writer.add_mesh(coords, conn) outside this for loop
            writer.add_field("coords", coords, time=t)

            # write the data
            writer.add_field("mult_with_t", result, time=t)

            # write a global quantity
            writer.add_global_field("sum", np.sum(result))

            # finish the step, required
            writer.finish_step()
