import onnx
import argparse
def modify(onnxpath,outpath):
    onnx_model = onnx.load(onnxpath)
    graph = onnx_model.graph
    i_name="images"
    imult=[1,3,1,1]
    for input_node in onnx_model.graph.input:
        i_name=input_node.name
        imult=[1,3,1,1]
        print("Before changing input size: {}".format(input_node.type.tensor_type.shape))
        if input_node.type.tensor_type.shape.dim[1].dim_value == 3:
            input_node.type.tensor_type.shape.dim[1].dim_value=1
        elif input_node.type.tensor_type.shape.dim[3].dim_value == 3:
            input_node.type.tensor_type.shape.dim[3].dim_value=1
            imult=[1,1,1,3]
            
    for input_node in onnx_model.graph.input:
        print("After changing input size: {}".format(input_node.type.tensor_type.shape))
    sub_const_node = onnx.helper.make_tensor(name='const_sub',
                      data_type=onnx.TensorProto.INT64,
                      dims=[4],
                      vals=imult)
    graph.initializer.append(sub_const_node)
    sub_node = onnx.helper.make_node(
        'Tile',
        name='Repeat',
        inputs=[str(i_name), 'const_sub'],
        outputs=['rpt'])
    graph.node.insert(0, sub_node)
    for _, node in enumerate(graph.node):
        if node.name =="Repeat":
            continue
        for i, input_node in enumerate(node.input):
            if str(i_name) == input_node:
                node.input[i] = 'rpt'
    info_model = onnx.helper.make_model(graph)
    onnx_model = onnx.shape_inference.infer_shapes(info_model)
    onnx.save(onnx_model, outpath)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',default="best.onnx")
    parser.add_argument('-o', '--output',default="best_modify.onnx")
    args = parser.parse_args()
    modify(args.input,args.output)

if __name__=="__main__":
    main()
    
