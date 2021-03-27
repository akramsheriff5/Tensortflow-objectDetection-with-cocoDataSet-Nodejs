const cocoSsd = require("@tensorflow-models/coco-ssd")
const tf = require("@tensorflow/tfjs-node")
const fs = require("fs").promises


// Load the Coco SSD model and image.
Promise.all([cocoSsd.load(), fs.readFile("./sample-images/image2.jpg")])
	.then((results) => {
		console.log(results)
		// First result is the COCO-SSD model object.
		const model = results[0]
		// Second result is image buffer.
		const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3)
		// Call detect() to run inference.
		return model.detect(imgTensor)
	})
	.then((predictions) => {
		console.log(JSON.stringify(predictions, null, 2))

  
	})
	.catch((err) => {
		console.log(err)
	})