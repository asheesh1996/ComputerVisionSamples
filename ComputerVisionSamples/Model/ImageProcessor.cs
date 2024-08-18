using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Bmp;
using SixLabors.ImageSharp.PixelFormats;
using Point = OpenCvSharp.Point;

public class ImageProcessor : IDisposable
{
    internal List<string> examplesNamesList => examplesMap.Keys.ToList();
    private readonly Dictionary<string, Func<byte[], byte[]>> examplesMap = new Dictionary<string, Func<byte[], byte[]>>();
    public int imageHeight;
    public int imageWidth;
    public ImageProcessor()
    {
        //examplesMap.Add("Test Loop", ImageLoop);
        examplesMap.Add("Canny Edge Detection", CannyEdgeDetection);
        //examplesMap.Add("Optical Flow", OpticalFlow);
        examplesMap.Add("Contour And Shape", ContoursAndShape);
        examplesMap.Add("K-Means", KMeans);
        examplesMap.Add("Gaussian Blur", GaussianBlur);
        examplesMap.Add("Median Blur", MedianBlur);
        //examplesMap.Add("Histogram", HistoGram);
        examplesMap.Add("ORB Features", ORBFeatures);
        //Console.WriteLine(examplesMap.Count);

    }

    public byte[] ExecuteExample(string exampleName, byte[] data)
    {
        if (examplesMap.TryGetValue(exampleName, out var exampleFunction))
        {
            return exampleFunction(data);
        }
        else
        {
            throw new ArgumentException("Example not found", nameof(exampleName));
        }
    }
    private Mat ConvertImageToMat(byte[] data)
    {
        using (var loadedImage = LoadImageFromByteArrayAsync(data))
        {
            var convertedData = ConvertImageToByteArrayAsync(loadedImage);
            return Mat.FromImageData(convertedData, ImreadModes.Color);
        }
    }
    public static Image<Rgba32> LoadImageFromByteArrayAsync(byte[] imageData)
    {
        using (var stream = new MemoryStream(imageData))
        {
            return Image.Load<Rgba32>(stream);
        }
    }
    public static byte[] ConvertImageToByteArrayAsync(Image<Rgba32> image)
    {
        using (var memoryStream = new MemoryStream())
        {
            image.Save(memoryStream, new BmpEncoder());
            return memoryStream.ToArray();
        }
    }
    private byte[] ImageLoop(byte[] data)
    {
        using (Mat img = ConvertImageToMat(data))
        {
            return img.ToBytes(".bmp");
        }
    }
    private byte[] CannyEdgeDetection(byte[] data)
    {
        using (Mat img = ConvertImageToMat(data))
        {
            using (Mat edges = new Mat())
            {
                Cv2.Canny(img, edges, 100, 200);
                return edges.ToBytes(".bmp");
            }
        }
    }
    Mat prevFrame = new Mat();
    private byte[] OpticalFlow(byte[] data)
    {
        using (Mat img = ConvertImageToMat(data))
        {
            Mat Gray = new Mat();
            Cv2.CvtColor(img, Gray, ColorConversionCodes.BGR2GRAY);
            if (prevFrame.Empty())
            {
                prevFrame = Gray.Clone();
            }
            Mat flow = new Mat();
            Cv2.CalcOpticalFlowFarneback(prevFrame, Gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
            prevFrame = Gray.Clone();
            return flow.ToBytes(".bmp");
        }
    }
    private byte[] ContoursAndShape(byte[] data)
    {
        using (Mat img = ConvertImageToMat(data))
        {
            Mat Gray = new Mat();
            Cv2.CvtColor(img, Gray, ColorConversionCodes.BGR2GRAY);
            using (Mat binary = new Mat())
            {
                Cv2.Threshold(Gray, binary, 128, 255, ThresholdTypes.Binary);
                Point[][] contours;
                HierarchyIndex[] hierarchy;
                Cv2.FindContours(binary, out contours, out hierarchy, RetrievalModes.Tree, ContourApproximationModes.ApproxSimple);
                using (Mat contourImg = new Mat(Gray.Size(), MatType.CV_8UC3))
                {
                    Cv2.DrawContours(contourImg, contours, -1, Scalar.Red, 2);
                    return contourImg.ToBytes(".bmp");
                }
            }
        }
    }
    private byte[] KMeans(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            Mat data = img.Reshape(1, img.Rows * img.Cols);
            data.ConvertTo(data, MatType.CV_32F);

            // Define criteria and number of clusters
            TermCriteria criteria = new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 100, 0.2);
            int k = 3; // Number of clusters

            // Mat to hold cluster centers and labels
            Mat labels = new Mat();
            Mat centers = new Mat();

            // Apply K-Means clustering
            Cv2.Kmeans(data, k, labels, criteria, 10, KMeansFlags.PpCenters, centers);

            // Convert centers to the same type as the original image
            centers.ConvertTo(centers, MatType.CV_8UC3);

            // Create a segmented image based on the labels
            Mat segmentedImg = new Mat(img.Size(), MatType.CV_8UC3);
            for (int i = 0; i < data.Rows; i++)
            {
                int clusterIdx = labels.At<int>(i);
                Vec3b color = centers.Row(clusterIdx).At<Vec3b>(0);
                segmentedImg.Set(i / img.Cols, i % img.Cols, color);
            }
            return segmentedImg.ToBytes(".bmp");
        }
    }
    private byte[] GaussianBlur(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            Mat gaussianBlurred = new Mat();
            Cv2.GaussianBlur(img, gaussianBlurred, new OpenCvSharp.Size(15, 15), 0);
            return gaussianBlurred.ToBytes(".bmp");
        }
    }
    private byte[] MedianBlur(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            Mat medianBlurred = new Mat();
            Cv2.MedianBlur(img, medianBlurred, 15);
            return medianBlurred.ToBytes(".bmp");

        }
    }
    private byte[] HistoGram(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            Mat equalizedImg = new Mat();
            Cv2.EqualizeHist(img, equalizedImg);
            return equalizedImg.ToBytes(".bmp");

        }
    }
    private byte[] ORBFeatures(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            Mat grayImg = new Mat();
            Cv2.CvtColor(img, grayImg, ColorConversionCodes.BGR2GRAY);

            // Detect ORB features
            ORB orb = ORB.Create();
            KeyPoint[] keyPoints;
            Mat descriptors = new Mat();
            orb.DetectAndCompute(grayImg, null, out keyPoints, descriptors);

            // Draw keypoints
            Mat keypointsImg = new Mat();
            Cv2.DrawKeypoints(img, keyPoints, keypointsImg);

            return keypointsImg.ToBytes(".bmp");

        }
    }
    public void Dispose()
    {
        prevFrame.Dispose();
        examplesMap.Clear();
    }
}
