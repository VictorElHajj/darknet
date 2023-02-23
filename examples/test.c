#include "darknet.h"
#include "convolutional_layer.h"
#include <stdio.h>
#ifdef NNPACK
#include "nnpack.h"
#endif


network create_network(layer l, float *data){
    network net = *make_network(1);
    net.layers[0] = l;
    net.workspace = calloc(1, l.workspace_size);
    
    
    net.outputs = l.outputs;

    net.inputs = l.inputs;
    net.input = data;

    return net;
}

void printIn(network net, convolutional_layer l){
    int i;
    int step = l.h;
    printf("Inputs: %d, \n w: %d \n", net.inputs, l.w);

    for (i = 0; i < net.inputs; i+=step){
        for(int j = 0; j < step; j++){
            printf("%f ", net.input[i+j]);
        }
        printf("\n");
    }
}


void printOut(network net, convolutional_layer l){
    int i;
    int step = l.out_h;
    printf("Outputs: %d \n", net.outputs);

    for (i = 0; i < net.outputs; i+=step){
        for(int j = 0; j < step; j++){

            printf("%f ", l.output[i+j]);
                        
        }
        printf("\n");
    }
}

int equal(const void *array_one, void *array_two, const size_t elem_count)
{
  return memcmp(array_one, array_two, elem_count * sizeof(float)) == 0;
}

int run_test(int argc, char **argv)
{

    float data[] = {
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,

        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 0, 0
    };

    float kernel[] = {
        0,1,0,
        0,1,0,
        0,1,0
    };

    float result[] = {
        0,0,1,1,0,0,
        0,0,1,1,0,0,
        0,0,1,1,0,0,

        0,0,1,1,0,0,
        0,0,1,1,0,0,
        0,0,1,1,0,0
    };


    convolutional_layer l = make_convolutional_layer(1, 8, 8, 1, 1, 1, 3, 1, 0, STAIR, 0, 0, 0, 0);
    l.weights = kernel;
 
    network net = create_network(l, data);

    printIn(net, l);

    forward_convolutional_layer(l , net);

    printOut(net, l);

    printf("Result: %d \n", equal(result, l.output, 36));


#ifdef NNPACK
    forward_convolutional_layer_nnp(l , net);
    
    printOut(net, l);

    printf("Result: %d \n", equal(result, l.output, 36));
#endif

    printf("Hello, World!\n");

    return 0;
}




