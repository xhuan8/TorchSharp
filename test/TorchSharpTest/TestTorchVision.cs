using System.Linq;
using static TorchSharp.torchvision.models;
using Xunit;
using System.IO;
using System;
using System.Collections.Generic;
using System.Threading;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;
using System.Reflection.Metadata;
using static TorchSharp.torch;
using static TorchSharp.torchvision.transforms;
using System.Xml;
using Modules.Detection;

namespace TorchSharp
{
#if NET472_OR_GREATER
    [Collection("Sequential")]
#endif // NET472_OR_GREATER
    public class TestTorchVision
    {
        [Fact]
        public void TestResNet18()
        {
            using var model = resnet18();
            var sd = model.state_dict();
            Assert.Equal(122, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet34()
        {
            using var model = resnet34();
            var sd = model.state_dict();
            Assert.Equal(218, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet50()
        {
            using var model = resnet50();
            var sd = model.state_dict();
            Assert.Equal(320, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet101()
        {
            using var model = resnet101();
            var sd = model.state_dict();
            Assert.Equal(626, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestResNet152()
        {
            using var model = resnet152();
            var sd = model.state_dict();
            Assert.Equal(932, sd.Count);

            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("bn1", names[1]),
                () => Assert.Equal("relu", names[2]),
                () => Assert.Equal("maxpool", names[3]),
                () => Assert.Equal("layer1", names[4]),
                () => Assert.Equal("layer2", names[5]),
                () => Assert.Equal("layer3", names[6]),
                () => Assert.Equal("layer4", names[7]),
                () => Assert.Equal("avgpool", names[8]),
                () => Assert.Equal("flatten", names[9]),
                () => Assert.Equal("fc", names[10])
            );
        }

        [Fact]
        public void TestAlexNet()
        {
            using var model = alexnet();
            var sd = model.state_dict();
            Assert.Equal(16, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("features", names[0]),
                () => Assert.Equal("avgpool", names[1]),
                () => Assert.Equal("classifier", names[2])
            );
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG11()
        {
            {
                using var model = vgg11();
                var sd = model.state_dict();
                Assert.Equal(22, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg11_bn();
                var sd = model.state_dict();
                Assert.Equal(62, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG13()
        {
            {
                using var model = vgg13();
                var sd = model.state_dict();
                Assert.Equal(26, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg13_bn();
                var sd = model.state_dict();
                Assert.Equal(76, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG16()
        {
            {
                using var model = vgg16();
                var sd = model.state_dict();
                Assert.Equal(32, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg16_bn();
                var sd = model.state_dict();
                Assert.Equal(97, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

#if DEBUG
        [Fact(Skip = "The test takes too long to run -- across the various VGG versions, 2/3 of overall test time is spent here.")]
#else
        [Fact]
#endif
        public void TestVGG19()
        {
            {
                using var model = vgg19();
                var sd = model.state_dict();
                Assert.Equal(38, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
            {
                using var model = vgg19_bn();
                var sd = model.state_dict();
                Assert.Equal(118, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("features", names[0]),
                    () => Assert.Equal("avgpool", names[1]),
                    () => Assert.Equal("classifier", names[2])
                );
            }
        }

        [Fact]
        public void TestInception()
        {
            using var model = inception_v3();
            var sd = model.state_dict();
            Assert.Equal(580, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("Conv2d_1a_3x3", names[0]),
                () => Assert.Equal("Conv2d_2a_3x3", names[1]),
                () => Assert.Equal("Conv2d_2b_3x3", names[2]),
                () => Assert.Equal("maxpool1", names[3]),
                () => Assert.Equal("Conv2d_3b_1x1", names[4]),
                () => Assert.Equal("Conv2d_4a_3x3", names[5]),
                () => Assert.Equal("maxpool2", names[6]),
                () => Assert.Equal("Mixed_5b", names[7]),
                () => Assert.Equal("Mixed_5c", names[8]),
                () => Assert.Equal("Mixed_5d", names[9]),
                () => Assert.Equal("Mixed_6a", names[10]),
                () => Assert.Equal("Mixed_6b", names[11]),
                () => Assert.Equal("Mixed_6c", names[12]),
                () => Assert.Equal("Mixed_6d", names[13]),
                () => Assert.Equal("Mixed_6e", names[14]),
                () => Assert.Equal("AuxLogits", names[15]),
                () => Assert.Equal("Mixed_7a", names[16]),
                () => Assert.Equal("Mixed_7b", names[17]),
                () => Assert.Equal("Mixed_7c", names[18]),
                () => Assert.Equal("avgpool", names[19]),
                () => Assert.Equal("dropout", names[20]),
                () => Assert.Equal("fc", names[21])
            );
        }

        [Fact]
        public void TestGoogLeNet()
        {
            using var model = googlenet();
            var sd = model.state_dict();
            Assert.Equal(344, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("conv1", names[0]),
                () => Assert.Equal("maxpool1", names[1]),
                () => Assert.Equal("conv2", names[2]),
                () => Assert.Equal("conv3", names[3]),
                () => Assert.Equal("maxpool2", names[4]),
                () => Assert.Equal("inception3a", names[5]),
                () => Assert.Equal("inception3b", names[6]),
                () => Assert.Equal("maxpool3", names[7]),
                () => Assert.Equal("inception4a", names[8]),
                () => Assert.Equal("inception4b", names[9]),
                () => Assert.Equal("inception4c", names[10]),
                () => Assert.Equal("inception4d", names[11]),
                () => Assert.Equal("inception4e", names[12]),
                () => Assert.Equal("maxpool4", names[13]),
                () => Assert.Equal("inception5a", names[14]),
                () => Assert.Equal("inception5b", names[15]),
                () => Assert.Equal("avgpool", names[16]),
                () => Assert.Equal("dropout", names[17]),
                () => Assert.Equal("fc", names[18])
            );
        }

        [Fact]
        public void TestMobileNetV2()
        {
            using var model = mobilenet_v2();
            var sd = model.state_dict();
            Assert.Equal(314, sd.Count);
            var names = model.named_children().Select(nm => nm.name).ToArray();
            Assert.Multiple(
                () => Assert.Equal("classifier", names[0]),
                () => Assert.Equal("features", names[1])
            );
        }

        [Fact]
        public void TestMobileNetV3()
        {
            using (var model = mobilenet_v3_large()) {
                var sd = model.state_dict();
                Assert.Equal(312, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("avgpool", names[0]),
                    () => Assert.Equal("classifier", names[1]),
                    () => Assert.Equal("features", names[2])
                );
            }

            using (var model = mobilenet_v3_small()) {
                var sd = model.state_dict();
                Assert.Equal(244, sd.Count);
                var names = model.named_children().Select(nm => nm.name).ToArray();
                Assert.Multiple(
                    () => Assert.Equal("avgpool", names[0]),
                    () => Assert.Equal("classifier", names[1]),
                    () => Assert.Equal("features", names[2])
                );
            }
        }

        [Fact]
        public void TestReadingAndWritingImages()
        {
            var fileName = "vslogo.jpg";
            var outName1 = $"TestReadingAndWritingImages_1_{fileName}";
            var outName2 = $"TestReadingAndWritingImages_2_{fileName}";

            if (System.IO.File.Exists(outName1)) System.IO.File.Delete(outName1);
            if (System.IO.File.Exists(outName2)) System.IO.File.Delete(outName2);

            torchvision.io.DefaultImager = new torchvision.io.SkiaImager(100);

            var img = torchvision.io.read_image(fileName);
            Assert.NotNull(img);
            Assert.Equal(torch.uint8, img.dtype);
            //Assert.Equal(new long[] { 3, 508, 728 }, img.shape);

            torchvision.io.write_image(img, outName1, torchvision.ImageFormat.Jpeg);
            Assert.True(System.IO.File.Exists(outName1));

            var img2 = torchvision.io.read_image(outName1);
            Assert.NotNull(img2);
            Assert.Equal(torch.uint8, img2.dtype);
            Assert.Equal(img.shape, img2.shape);

            var grey = torchvision.transforms.functional.rgb_to_grayscale(img);
            Assert.Equal(torch.float32, grey.dtype);

            torchvision.io.write_jpeg(torchvision.transforms.functional.convert_image_dtype(grey, torch.ScalarType.Byte), outName2);
            Assert.True(System.IO.File.Exists(outName2));

            System.IO.File.Delete(outName1);
            System.IO.File.Delete(outName2);
        }

        [Fact]
        public void TestFasterRCNNTrain()
        {
            torchvision.io.DefaultImager = new torchvision.io.SkiaImager();

            var train_dataset = new CustomDataset(Config.TraingDir, Config.ImageWidth, Config.ImageHeight,
                    Config.Classes, null);
            var valid_dataset = new CustomDataset(Config.EvalDir, Config.ImageWidth, Config.ImageHeight,
                    Config.Classes, null);
            var train_loader = new DataLoader<(Tensor, Dictionary<string, Tensor>), (IEnumerable<Tensor>, IEnumerable<Dictionary<string, Tensor>>)>(train_dataset, Config.BatchSize, FasterRCNNUtils.collate_fn, shuffle: true);
            var valid_loader = new DataLoader<(Tensor, Dictionary<string, Tensor>), (IEnumerable<Tensor>, IEnumerable<Dictionary<string, Tensor>>)>(valid_dataset, Config.BatchSize, FasterRCNNUtils.collate_fn, shuffle: false);
            Console.WriteLine(string.Format("Number of training samples: {0}", train_dataset.Count));
            Console.WriteLine(string.Format("Number of validation samples: {0}\n", valid_dataset.Count));

            // initialize the model and move to the computation device
            Dictionary<string, object> kwargs = new Dictionary<string, object>();
            kwargs.Add("max_size", 10000);
            kwargs.Add("box_detections_per_img", 300);
            kwargs.Add("rpn_pre_nms_top_n_test", 4000);
            kwargs.Add("rpn_post_nms_top_n_test", 3000);
            kwargs.Add("rpn_pre_nms_top_n_train", 4000);
            kwargs.Add("rpn_post_nms_top_n_train", 3000);
            kwargs.Add("box_nms_thresh", 0.2f);
            kwargs.Add("box_fg_iou_thresh", 0.7f);
            kwargs.Add("box_bg_iou_thresh", 0.3f);
            var model = torchvision.models.detection.fasterrcnn_resnet50_fpn("fasterrcnn_resnet50_fpn_coco-258fb6c6.pth", kwargs: kwargs);
            var load_check_point = false;
            if (load_check_point && File.Exists("outputs/best_model.pth")) {
                model.load("outputs/best_model.pth");
            }
            model = model.to(Config.Device);
            // get the model parameters
            var parameters = model.parameters().Where(p => p.requires_grad);
            // define the optimizer
            var optimizer = torch.optim.SGD(parameters, learningRate: 0.001, momentum: 0.9, weight_decay: 0.0005);

            // initialize the Averager class
            var train_loss_hist = new Averager();
            var val_loss_hist = new Averager();
            // train and validation loss lists to store loss values of all...
            // ... iterations till ena and plot graphs for all iterations
            var train_loss_list = new List<float>();
            var val_loss_list = new List<float>();

            // initialize SaveBestModel class
            var save_best_model = new SaveBestModel();

            // start the training epochs
            for (int epoch = 0; epoch < Config.NumberEpochs; epoch++) {
                Console.WriteLine(string.Format("\nEPOCH {0} of {1}", epoch + 1, Config.NumberEpochs));

                // reset the training and validation loss histories for the current epoch
                train_loss_hist.reset();
                val_loss_hist.reset();

                // start timer and carry out training and validation
                var start = DateTime.Now;

                var train_loss = FasterRCNNUtils.train(train_loader, model, optimizer, train_loss_hist);
                var val_loss = FasterRCNNUtils.validate(valid_loader, model, val_loss_hist);
                Console.WriteLine(string.Format("Epoch #{0} train loss: {1:.3f}", epoch + 1,
                    train_loss_hist.Value));
                Console.WriteLine(string.Format("Epoch #{0} validation loss: {1:.3f}",
                    epoch + 1, val_loss_hist.Value));
                var end = DateTime.Now;

                Console.WriteLine(string.Format("Took {0} minutes for epoch {1}",
                    (end - start).Seconds, epoch));
                Console.WriteLine(string.Format("Best Epoch #{0} validation loss: {1:.3f}",
                    save_best_model.Best_valid_epoch, save_best_model.Best_valid_loss));

                // save the best model till now if we have the least loss in the...
                // ... current epoch
                save_best_model.__call__(val_loss_hist.Value, epoch, model, optimizer);

                // sleep for 5 seconds after each epoch
                Thread.Sleep(5);
            }
        }
    }
}
