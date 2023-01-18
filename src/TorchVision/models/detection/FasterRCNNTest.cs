using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Serialization;
using System.Xml;
using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.utils.data;
using static TorchSharp.torchvision;
using static TorchSharp.torchvision.transforms;
using static TorchSharp.torch.optim;
using System.Security.Cryptography;
using TorchSharp.Modules.Detection;

namespace Modules.Detection
{
    public class Config
    {
        public static int BatchSize = 1;
        public static int ImageWidth = 5120;
        public static int ImageHeight = 5120;
        public static int NumberEpochs = 300;
        public static int NumberWorkers = 0;

        //public static torch.Device Device = torch.cuda.is_available() ? torch.CUDA : torch.CPU;
        public static torch.Device Device = torch.CPU;
        public static string TraingDir = "D:\\Document\\PinInspection\\PinImages1\\augment";
        public static string EvalDir = "D:\\Document\\PinInspection\\PinImages1\\eval";
        public static string TestDir = "D:\\Document\\PinInspection\\PinImages1\\test";

        public static List<string> Classes = new List<string> { "__background__", "S", "MH", "MV", "BH", "BV" };
        public static List<Color> Colors = new List<Color> { Color.FromArgb(0, 0, 0),
            Color.FromArgb(0,0,255), Color.FromArgb(0,215,255), Color.FromArgb(208,224,64),
            Color.FromArgb(173,222,255), Color.FromArgb(255,0,255)};
        public static int NumberClasses = Classes.Count;
        public static bool VisualizeTransformedImage = false;
        public static string OutDir = "outputs";
        public static string OutDirTest = "outputsTest";
    }

    public class CustomDataset : Dataset<(Tensor, Dictionary<string, Tensor>)>
    {
        private ITransform transforms;
        private string dir_path;
        private int height;
        private int width;
        private List<string> classes;
        private string[] image_paths;
        private string[] all_images;

        public override long Count { get => all_images.Length; }

        public CustomDataset(string dir_path, int width, int height, List<string> classes, ITransform transforms = null)
        {
            this.transforms = transforms;
            this.dir_path = dir_path;
            this.height = height;
            this.width = width;
            this.classes = classes;

            // get all the image paths in sorted order
            this.image_paths = Directory.GetFiles(this.dir_path, "*.png");
            this.all_images = image_paths.Select(p => Path.GetFileName(p)).ToArray();
            Array.Sort(this.all_images);
        }

        public override (Tensor, Dictionary<string, Tensor>) GetTensor(long idx)
        {
            // capture the image name and the full image path
            var image_name = this.all_images[idx];
            var image_path = Path.Combine(this.dir_path, image_name);

            // read the image
            var image = torchvision.io.read_image(image_path);
            // convert BGR to RGB color format
            var index = torch.LongTensor(new long[] { 2, 1, 0 });
            image[index] = image.clone();
            image = image.to(ScalarType.Float32);
            var image_resized = functional.resize(image, this.height, this.width);
            image_resized /= 255.0;

            // capture the corresponding XML file for getting the annotations
            string annot_filename = image_name.Replace(".png", ".xml");
            var annot_file_path = Path.Combine(this.dir_path, annot_filename);

            var labels = new List<int>();
            Annotation anno = LoadXML(annot_file_path, typeof(Annotation)) as Annotation;
            var boxes = new double[anno.Object.Count, 4];

            // get the height and width of the image
            var image_width = image.shape[1];
            var image_height = image.shape[0];

            // box coordinates for xml files are extracted and corrected for image size given
            for (int i = 0; i < anno.Object.Count; i++) {
                var member = anno.Object[i];
                // map the current object name to `classes` list to get...
                // ... the label index and append to `labels` list
                labels.Add(this.classes.IndexOf(member.Name));

                // xmin = left corner x-coordinates
                var xmin = int.Parse(member.Bndbox.Xmin);
                // xmax = right corner x-coordinates
                var xmax = int.Parse(member.Bndbox.Xmax);
                // ymin = left corner y-coordinates
                var ymin = int.Parse(member.Bndbox.Ymin);
                // ymax = right corner y-coordinates
                var ymax = int.Parse(member.Bndbox.Ymax);

                // resize the bounding boxes according to the...
                // ... desired `width`, `height`
                var xmin_final = ((double)xmin / image_width) * this.width;
                var xmax_final = ((double)xmax / image_width) * this.width;
                var ymin_final = ((double)ymin / image_height) * this.height;
                var yamx_final = ((double)ymax / image_height) * this.height;

                boxes[i, 0] = xmin_final;
                boxes[i, 1] = xmax_final;
                boxes[i, 2] = ymin_final;
                boxes[i, 3] = yamx_final;
            }

            // bounding box to tensor
            var boxes_tensor = torch.as_tensor(boxes, dtype: torch.float32);
            // area of the bounding boxes
            var area = (boxes_tensor[TensorIndex.Colon, 3] - boxes_tensor[TensorIndex.Colon, 1]) *
                (boxes_tensor[TensorIndex.Colon, 2] - boxes_tensor[TensorIndex.Colon, 0]);
            // no crowd instances
            var iscrowd = torch.zeros(new long[] { boxes_tensor.shape[0] }, dtype: torch.int64);
            // labels to tensor
            var labels_tensor = torch.as_tensor(labels, dtype: torch.int64);

            // prepare the final `target` dictionary
            var target = new Dictionary<string, Tensor>();
            target["boxes"] = boxes_tensor;
            target["labels"] = labels_tensor;
            target["area"] = area;
            target["iscrowd"] = iscrowd;
            var image_id = torch.tensor(new long[] { idx });
            target["image_id"] = image_id;

            // apply the image transforms
            if (this.transforms != null)
                image_resized = this.transforms.forward(image_resized);

            return (image_resized, target);
        }

        public static object LoadXML(string fileName, Type objectType)
        {
            object obj = null;

            if (File.Exists(fileName)) {
                XmlReaderSettings settings = new XmlReaderSettings();
                settings.IgnoreComments = true;
                settings.IgnoreWhitespace = true;
                settings.IgnoreProcessingInstructions = true;

                using (XmlReader reader = XmlReader.Create(fileName, settings)) {
                    XmlSerializer ser = new XmlSerializer(objectType);
                    obj = ser.Deserialize(reader);
                }
            }
            return obj;
        }
    }

    // PascalVoc
    [XmlRoot(ElementName = "source")]
    public class Source
    {
        [XmlElement(ElementName = "database")]
        public string Database { get; set; }
    }

    [XmlRoot(ElementName = "size")]
    public class Size
    {
        [XmlElement(ElementName = "width")]
        public string Width { get; set; }
        [XmlElement(ElementName = "height")]
        public string Height { get; set; }
        [XmlElement(ElementName = "depth")]
        public string Depth { get; set; }
    }

    [XmlRoot(ElementName = "bndbox")]
    public class Bndbox
    {
        [XmlElement(ElementName = "xmin")]
        public string Xmin { get; set; }
        [XmlElement(ElementName = "xmax")]
        public string Xmax { get; set; }
        [XmlElement(ElementName = "ymin")]
        public string Ymin { get; set; }
        [XmlElement(ElementName = "ymax")]
        public string Ymax { get; set; }
    }

    [XmlRoot(ElementName = "object")]
    public class Object
    {
        [XmlElement(ElementName = "name")]
        public string Name { get; set; }
        [XmlElement(ElementName = "pose")]
        public string Pose { get; set; }
        [XmlElement(ElementName = "truncated")]
        public string Truncated { get; set; }
        [XmlElement(ElementName = "difficult")]
        public string Difficult { get; set; }
        [XmlElement(ElementName = "occluded")]
        public string Occluded { get; set; }
        [XmlElement(ElementName = "bndbox")]
        public Bndbox Bndbox { get; set; }
    }

    [XmlRoot(ElementName = "annotation")]
    public class Annotation
    {
        [XmlElement(ElementName = "folder")]
        public string Folder { get; set; }
        [XmlElement(ElementName = "filename")]
        public string Filename { get; set; }
        [XmlElement(ElementName = "path")]
        public string Path { get; set; }
        [XmlElement(ElementName = "source")]
        public Source Source { get; set; }
        [XmlElement(ElementName = "size")]
        public Size Size { get; set; }
        [XmlElement(ElementName = "segmented")]
        public string Segmented { get; set; }
        [XmlElement(ElementName = "object")]
        public List<Object> Object { get; set; }
    }

    // utils
    public class Averager
    {
        private float current_total;
        private int iterations;

        public float Value {
            get {
                if (this.iterations == 0)
                    return 0;
                else
                    return this.current_total / this.iterations;
            }
        }

        public Averager()
        {
            this.current_total = 0.0f;
            this.iterations = 0;
        }

        public void send(float value)
        {
            this.current_total += value;
            this.iterations += 1;
        }

        public void reset()
        {
            this.current_total = 0.0f;
            this.iterations = 0;
        }
    }

    /// <summary>
    /// Class to save the best model while training. If the current epoch's 
    /// validation loss is less than the previous least less, then save the
    /// model state.
    /// </summary>
    public class SaveBestModel
    {
        private float best_valid_loss;
        private int best_valid_epoch;

        public SaveBestModel(float best_valid_loss = float.PositiveInfinity)
        {
            this.best_valid_loss = best_valid_loss;
            this.best_valid_epoch = 0;
        }

        public int Best_valid_epoch { get => best_valid_epoch; }
        public float Best_valid_loss { get => best_valid_loss; }

        public void __call__(float current_valid_loss, int epoch, nn.Module model, Optimizer optimizer)
        {
            if (current_valid_loss < this.Best_valid_loss) {
                this.best_valid_loss = current_valid_loss;
                this.best_valid_epoch = epoch + 1;
                Console.WriteLine(string.Format("\nBest validation loss: {0}", this.Best_valid_loss));
                Console.WriteLine(string.Format("\nSaving best model for epoch: {0}\n", this.Best_valid_epoch));
                model.save("outputs/best_model.pth");
            }
        }
    }

    public class FasterRCNNUtils
    {
        public static (IEnumerable<Tensor>, IEnumerable<Dictionary<string, Tensor>>) collate_fn(IEnumerable<(Tensor, Dictionary<string, Tensor>)> input, torch.Device device)
        {
            List<Tensor> list = new List<Tensor>();
            List<Dictionary<string, Tensor>> targets = new List<Dictionary<string, Tensor>>();
            foreach (var item in input) {
                list.Add(item.Item1.to(device));
                Dictionary<string, Tensor> t = new Dictionary<string, Tensor>();
                foreach (var keyPair in item.Item2) {
                    t.Add(keyPair.Key, keyPair.Value.to(device));
                }
                targets.Add(t);
            }

            return (list, targets);
        }

        public static List<float> train(DataLoader<(Tensor, Dictionary<string, Tensor>), (IEnumerable<Tensor>, IEnumerable<Dictionary<string, Tensor>>)> train_data_loader,
            FasterRCNN model, Optimizer optimizer, Averager train_loss_hist)
        {
            Console.WriteLine("Training");
            int train_itr = 0;
            var train_loss_list = new List<float>();

            var loader = train_data_loader.GetEnumerator();
            for (int i = 0; i < train_data_loader.Count; i++) {
                loader.MoveNext();
                var data = loader.Current;

                optimizer.zero_grad();
                var (images, targets) = data;

                var loss_dict = model.forward(images.ToList(), targets.ToList());
                var values = loss_dict.Item1.Values.ToArray();
                var losses = torch.stack(values).sum(dim: 0);
                var loss_value = losses.item<float>();
                train_loss_list.Add(loss_value);

                train_loss_hist.send(loss_value);

                losses.backward();
                optimizer.step();

                train_itr += 1;
            }

            return train_loss_list;
        }

        public static List<float> validate(DataLoader<(Tensor, Dictionary<string, Tensor>), (IEnumerable<Tensor>, IEnumerable<Dictionary<string, Tensor>>)> valid_data_loader,
            FasterRCNN model, Averager val_loss_hist)
        {
            // function for running validation iterations
            Console.WriteLine("Validating");
            int val_itr = 0;
            List<float> val_loss_list = new List<float>();

            var loader = valid_data_loader.GetEnumerator();
            for (int i = 0; i < valid_data_loader.Count; i++) {
                loader.MoveNext();
                var data = loader.Current;

                var (images, targets) = data;

                (Dictionary<string, Tensor>, List<Dictionary<string, Tensor>>) loss_dict;
                using (torch.no_grad())
                    loss_dict = model.forward(images.ToList(), targets.ToList());

                var values = loss_dict.Item1.Values.ToArray();
                var losses = torch.stack(values).sum(dim: 0);
                var loss_value = losses.item<float>();
                val_loss_list.Add(loss_value);

                val_loss_hist.send(loss_value);
                val_itr += 1;
            }
            return val_loss_list;
        }
    }
}
