import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class Facerecognition {

    public static void main(String[] args) {

        // 1️⃣ Load OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        // 2️⃣ Load Haar Cascade file
        String cascadePath =
                "C:/Users/ASUS/OneDrive/Desktop/project/haarcascade_frontalface_default.xml";


        CascadeClassifier faceDetector = new CascadeClassifier(cascadePath);

        // 3️⃣ Check if cascade loaded
        if (faceDetector.empty()) {
            System.out.println("ERROR: Haar cascade NOT loaded!");
            return;
        } else {
            System.out.println("Haar cascade loaded successfully.");
        }

        // 4️⃣ Open webcam
        VideoCapture camera = new VideoCapture(0);

        if (!camera.isOpened()) {
            System.out.println("ERROR: Camera not available!");
            return;
        }

        Mat frame = new Mat();

        // 5️⃣ Read frames
        while (true) {
            camera.read(frame);
            if (frame.empty()) break;

            Mat gray = new Mat();
            Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);

            MatOfRect faces = new MatOfRect();
            faceDetector.detectMultiScale(gray, faces);

            // 6️⃣ Draw rectangles around faces
            for (Rect rect : faces.toArray()) {
                Imgproc.rectangle(
                        frame,
                        new Point(rect.x, rect.y),
                        new Point(rect.x + rect.width, rect.y + rect.height),
                        new Scalar(0, 255, 0),
                        2
                );
            }

            HighGui.imshow("Face Detection", frame);

            // Press ESC to exit
            if (HighGui.waitKey(30) == 27) {
                break;
            }
        }

        // 7️⃣ Release resources
        camera.release();
        HighGui.destroyAllWindows();
    }
}
