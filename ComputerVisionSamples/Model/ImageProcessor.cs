using Microsoft.AspNetCore.Components.Forms;
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
        examplesMap.Add("Detect Circles", DetectCircles);
        //examplesMap.Add("DFT", DFTPowerSpectrum);
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
    private static Mat ConvertImageToMat(byte[] data)
    {
        using (var stream = new MemoryStream(data))
        {
            // Load the image from the byte array
            using (var loadedImage = Image.Load<Rgba32>(stream))
            {
                using (var memoryStream = new MemoryStream())
                {
                    // Convert the loaded image to a byte array in BMP format
                    loadedImage.Save(memoryStream, new BmpEncoder());
                    byte[] convertedData = memoryStream.ToArray();

                    // Return as a Mat
                    return Mat.FromImageData(convertedData, ImreadModes.Color);
                }
            }
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
            using (Mat edges = new Mat(img.Size(), MatType.CV_8UC3, Scalar.All(0)))
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
            using (Mat flow = new Mat(img.Size(), MatType.CV_8UC3, Scalar.All(0)))
            {
                Cv2.CalcOpticalFlowFarneback(prevFrame, Gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
                prevFrame = Gray.Clone();
                return flow.ToBytes(".bmp");
            }
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
                using (Mat contourImg = new Mat(img.Size(), MatType.CV_8UC3, Scalar.All(0)))
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
            using (Mat segmentedImg = new Mat(img.Size(), MatType.CV_8UC3))
            {
                for (int i = 0; i < data.Rows; i++)
                {
                    int clusterIdx = labels.At<int>(i);
                    Vec3b color = centers.Row(clusterIdx).At<Vec3b>(0);
                    segmentedImg.Set(i / img.Cols, i % img.Cols, color);
                }
                return segmentedImg.ToBytes(".bmp");
            }
        }
    }
    private byte[] GaussianBlur(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            using (Mat gaussianBlurred = new Mat(img.Size(), MatType.CV_8UC3))
            {
                Cv2.GaussianBlur(img, gaussianBlurred, new OpenCvSharp.Size(15, 15), 0);
                return gaussianBlurred.ToBytes(".bmp");
            }
        }
    }
    private byte[] MedianBlur(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            using (Mat medianBlurred = new Mat(img.Size(), MatType.CV_8UC3))
            {
                Cv2.MedianBlur(img, medianBlurred, 15);
                return medianBlurred.ToBytes(".bmp");
            }
        }
    }

    private byte[] DetectCircles(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            Mat gray = new Mat();
            Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
            // Apply median blur to the grayscale image
            Cv2.MedianBlur(gray, gray, 5);
            // Vector to store detected circles
            var circles = Cv2.HoughCircles(
                gray,
                HoughModes.Gradient,
                1,
                gray.Rows / 16, // Minimum distance between circles
                100,           // Higher threshold for Canny edge detector
                30,            // Lower threshold for center detection
                1,             // Minimum circle radius
                30             // Maximum circle radius
            );
            // Draw the detected circles on the original image
            foreach (var c in circles)
            {
                var center = c.Center.ToPoint();
                // Circle center
                Cv2.Circle(img, center, 1, new Scalar(0, 100, 100), 3, LineTypes.AntiAlias);
                // Circle outline
                Cv2.Circle(img, center, (int)c.Radius, new Scalar(255, 0, 255), 3, LineTypes.AntiAlias);
            }
            return img.ToBytes(".bmp");
        }
    }
    private byte[] HistoGram(byte[] inputdata)
    {
        using (Mat img = ConvertImageToMat(inputdata))
        {
            using (Mat equalizedImg = new Mat(img.Size(), MatType.CV_8UC3))
            {
                Cv2.EqualizeHist(img, equalizedImg);
                return equalizedImg.ToBytes(".bmp");
            }
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

            using (Mat keypointsImg = new Mat(img.Size(), MatType.CV_8UC3))
            {
                Cv2.DrawKeypoints(img, keyPoints, keypointsImg);
                return keypointsImg.ToBytes(".bmp");
            }

        }
    }
    private byte[] DFTPowerSpectrum(byte[] inputdata)
    {
        using (Mat inputImage = ConvertImageToMat(inputdata))
        {
            // Expand input image to optimal size
            Mat padded = new Mat();
            int m = Cv2.GetOptimalDFTSize(inputImage.Rows);
            int n = Cv2.GetOptimalDFTSize(inputImage.Cols);
            Cv2.CopyMakeBorder(inputImage, padded, 0, m - inputImage.Rows, 0, n - inputImage.Cols, BorderTypes.Constant, new Scalar(0));

            // Create a matrix with two channels (real and imaginary)
            Mat[] planes = { new Mat(padded.Size(), MatType.CV_32F), new Mat(padded.Size(), MatType.CV_32F) };
            Mat complexI = new Mat();
            Cv2.Merge(planes, complexI);

            // Perform DFT
            Cv2.Dft(complexI, complexI);

            // Compute the magnitude and switch to logarithmic scale
            Cv2.Split(complexI, out planes);
            Cv2.Magnitude(planes[0], planes[1], planes[0]);
            Mat magI = planes[0];

            magI += new Scalar(1); // Switch to logarithmic scale
            Cv2.Log(magI, magI);

            // Crop the spectrum, if it has an odd number of rows or columns
            magI = magI[new Rect(0, 0, magI.Cols & -2, magI.Rows & -2)];

            // Rearrange the quadrants of the Fourier image so that the origin is at the image center
            int cx = magI.Cols / 2;
            int cy = magI.Rows / 2;

            Mat q0 = magI[new Rect(0, 0, cx, cy)]; // Top-Left
            Mat q1 = magI[new Rect(cx, 0, cx, cy)]; // Top-Right
            Mat q2 = magI[new Rect(0, cy, cx, cy)]; // Bottom-Left
            Mat q3 = magI[new Rect(cx, cy, cx, cy)]; // Bottom-Right

            // Swap quadrants
            Mat tmp = new Mat();
            q0.CopyTo(tmp);
            q3.CopyTo(q0);
            tmp.CopyTo(q3);

            q1.CopyTo(tmp); // Swap quadrant (Top-Right with Bottom-Left)
            q2.CopyTo(q1);
            tmp.CopyTo(q2);

            // Normalize the magnitude image for display
            Cv2.Normalize(magI, magI, 0, 1, NormTypes.MinMax);

            return magI.ToBytes(".bmp"); ; // Return the power spectrum
        }
    }
    public void Dispose()
    {
        prevFrame.Dispose();
        examplesMap.Clear();
    }
}
