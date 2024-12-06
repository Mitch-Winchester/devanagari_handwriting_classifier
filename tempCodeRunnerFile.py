if i % 10 == 0:
            print("Epoch: ", i)
            predictions = get_predictions(A2)
            print("Accuracy: ",get_accuracy(predictions, Y[BATCH_SIZE*j:BATCH_SIZE*(j+1)]))