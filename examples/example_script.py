import numpy as np
import argparse

from dbmanager_v2.simulation import SimulationWriter



def main(path, uid):

    # initiate the database simulation writer
    writer = SimulationWriter(uid, path)
    writer.register_git_attributes()
    writer.add_metadata()

    # to load the parameters of this simulation use
    parameters = writer.parameters
    
    # do what you have to do
    # here we create a simple mesh
    coords = np.array([
        [0., 0.],
        [1., 0.],
        [2., 0.],
        [0., 1.],
        [1., 2.]
    ])
    
    initial_values = np.random.randint(0, 10, 5)
    
    for t in range(parameters['timesteps']):

        result = initial_values * t**parameters['exponent']
        
        # write the coordinates
        # if you have a mesh with coords and connectivity, you can add a global mesh with
        # writer.add_mesh(coords, conn) outside this for loop
        writer.add_field('coords', coords, time=t)
        
        # write the data
        writer.add_field('mult_with_t', result, time=t)

        # write a global quantity
        writer.add_global_field('sum', np.sum(result))
        
        # finish the step, required
        writer.finish_step()

    # finish sim, not required
    writer.finish_sim()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--uid', type=str)
    args = parser.parse_args()

    main(args.path, args.uid)
