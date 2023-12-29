using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.CV;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Runtime.InteropServices;
using System.Globalization;
namespace JakaEyeInHandCalibration
{
    public static class MatExtension
    {
        public static dynamic GetValue(this Mat mat, int row, int col)
        {
            var value = CreateElement(mat.Depth);
            Marshal.Copy(mat.DataPointer + (row * mat.Cols + col) * mat.ElementSize, value, 0, 1);
            return value[0];
        }

        public static void SetValue(this Mat mat, int row, int col, dynamic value)
        {
            var target = CreateElement(mat.Depth, value);
            Marshal.Copy(target, 0, mat.DataPointer + (row * mat.Cols + col) * mat.ElementSize, 1);
        }
        private static dynamic CreateElement(DepthType depthType, dynamic value)
        {
            var element = CreateElement(depthType);
            element[0] = value;
            return element;
        }

        private static dynamic CreateElement(DepthType depthType)
        {
            if (depthType == DepthType.Cv8S)
            {
                return new sbyte[1];
            }
            if (depthType == DepthType.Cv8U)
            {
                return new byte[1];
            }
            if (depthType == DepthType.Cv16S)
            {
                return new short[1];
            }
            if (depthType == DepthType.Cv16U)
            {
                return new ushort[1];
            }
            if (depthType == DepthType.Cv32S)
            {
                return new int[1];
            }
            if (depthType == DepthType.Cv32F)
            {
                return new float[1];
            }
            if (depthType == DepthType.Cv64F)
            {
                return new double[1];
            }
            return new float[1];
        }
    }

    public class PositionRotationData
    {
        public double X, Y, Z;
        public double Rx, Ry, Rz;

        public PositionRotationData(double x, double y, double z, double rx, double ry, double rz)
        {
            X = x;
            Y = y;
            Z = z;
            Rx = rx;
            Ry = ry;
            Rz = rz;
        }
    }

    public class CalibrateHandEye
    {
        #region Properties
        private string _chessboardImageFolderPath { get; set; }
        private string _robotDataPath { get; set; }
        private string _drawChessboardFolderPath { get; set; }
        private Size _boardSize { get; set; }
        private float _squareSize { get; set; }
        private VectorOfPoint3D32F _objectPoints { get; set; } = new VectorOfPoint3D32F();
        private string _rotationSequence;
        public string RotationSequence
        {
            get
            {
                return _rotationSequence;
            }
            set
            {
                if (value != "zyx" && value != "yzx" && value != "zxy" && value != "xzy" && value != "yxz" && value != "xyz")
                {
                    throw new System.ArgumentException("Rotation Sequence must be zyx, yzx, zxy, xzy, yxz, xyz");
                }

                _rotationSequence = value;
            }
        }
        public string DebugMessage { get; set; } = "";
        public List<Matrix<double>> RobotPoseTransformationMatrix { get; set; }
        public List<Matrix<double>> CameraPoseTransformationMatrix { get; set; }
        public Matrix<double> TransformationMatrixX { get; set; }
        #endregion


        #region Constructor
        public CalibrateHandEye(string chessboardImageFolderPath, string robotDataPath, string drawChessboardFolderPath, int boardWidth, int boardHeight, float squareSize, string rotationSequence)
        {
            _chessboardImageFolderPath = chessboardImageFolderPath;
            _robotDataPath = robotDataPath;
            _drawChessboardFolderPath = drawChessboardFolderPath;
            _boardSize = new Size(boardWidth, boardHeight);
            _squareSize = squareSize;
            for (int height = 0; height < boardHeight; height++)
            {
                for (int width = 0; width < boardWidth; width++)
                {
                    _objectPoints.Push(new[] { new MCvPoint3D32f(width * squareSize, height * squareSize, 0) });
                }
            }
            _rotationSequence = rotationSequence;
        }
        #endregion

        #region Check info
        public string GetImagePath()
        {
            return _chessboardImageFolderPath;
        }

        public string GetRobotDataPath()
        {
            return _robotDataPath;
        }

        #endregion

        public bool IsRotationMatrix(Matrix<double> rotationMatrix)
        {
            /*DebugMessage += $"Rotation Matrix\n" +
                            $"{rotationMatrix[0, 0]} {rotationMatrix[0, 1]} {rotationMatrix[0, 2]}\n" +
                            $"{rotationMatrix[1, 0]} {rotationMatrix[1, 1]} {rotationMatrix[1, 2]}\n" +
                            $"{rotationMatrix[2, 0]} {rotationMatrix[2, 1]} {rotationMatrix[2, 2]}\n";
            Matrix<double> transposeMatrix = rotationMatrix.Transpose();
            DebugMessage += $"Transpose Matrix\n" +
                            $"{transposeMatrix[0, 0]} {transposeMatrix[0, 1]} {transposeMatrix[0, 2]}\n" +
                            $"{transposeMatrix[1, 0]} {transposeMatrix[1, 1]} {transposeMatrix[1, 2]}\n" +
                            $"{transposeMatrix[2, 0]} {transposeMatrix[2, 1]} {transposeMatrix[2, 2]}\n";*/

            Matrix<double> shouldBeIdentity;
            shouldBeIdentity = rotationMatrix.Transpose() * rotationMatrix;
            /*DebugMessage += $"Should Be Identity Matrix\n" +
                            $"{shouldBeIdentity[0, 0]} {shouldBeIdentity[0, 1]} {shouldBeIdentity[0, 2]}\n" +
                            $"{shouldBeIdentity[1, 0]} {shouldBeIdentity[1, 1]} {shouldBeIdentity[1, 2]}\n" +
                            $"{shouldBeIdentity[2, 0]} {shouldBeIdentity[2, 1]} {shouldBeIdentity[2, 2]}\n";*/

            Matrix<double> I = new Matrix<double>(3, 3);
            I.SetIdentity();
            /*DebugMessage += $"Identity Matrix\n" +
                            $"{I[0, 0]} {I[0, 1]} {I[0, 2]}\n" +
                            $"{I[1, 0]} {I[1, 1]} {I[1, 2]}\n" +
                            $"{I[2, 0]} {I[2, 1]} {I[2, 2]}\n";*/

            /*DebugMessage += $"Norm = {CvInvoke.Norm(I, shouldBeIdentity)}";*/
            return CvInvoke.Norm(I, shouldBeIdentity) < 1e-6;
        }

        public Matrix<double> EulerAngleToRotationmatrix(double rx, double ry, double rz)
        {
            //the rx, ry, rz is in radian unit
            var sineRx = Math.Sin(rx); var cosRx = Math.Cos(rx);
            var sineRy = Math.Sin(ry); var cosRy = Math.Cos(ry);
            var sineRz = Math.Sin(rz); var cosRz = Math.Cos(rz);

            Matrix<double> rotationX = new Matrix<double>(3, 3);
            rotationX[0, 0] = 1; rotationX[0, 1] = 0; rotationX[0, 2] = 0;
            rotationX[1, 0] = 0; rotationX[1, 1] = cosRx; rotationX[1, 2] = -sineRx;
            rotationX[2, 0] = 0; rotationX[2, 1] = sineRx; rotationX[2, 2] = cosRx;

            Matrix<double> rotationY = new Matrix<double>(3, 3);
            rotationY[0, 0] = cosRy; rotationY[0, 1] = 0; rotationY[0, 2] = sineRy;
            rotationY[1, 0] = 0; rotationY[1, 1] = 1; rotationY[1, 2] = 0;
            rotationY[2, 0] = -sineRy; rotationY[2, 1] = 0; rotationY[2, 2] = cosRy;

            Matrix<double> rotationZ = new Matrix<double>(3, 3);
            rotationZ[0, 0] = cosRz; rotationZ[0, 1] = -sineRz; rotationZ[0, 2] = 0;
            rotationZ[1, 0] = sineRz; rotationZ[1, 1] = cosRz; rotationZ[1, 2] = 0;
            rotationZ[2, 0] = 0; rotationZ[2, 1] = 0; rotationZ[2, 2] = 1;

            Matrix<double> rotationMatrix = new Matrix<double>(3, 3);
            if (_rotationSequence == "zyx")
            {
                rotationMatrix = rotationX * rotationY * rotationZ;
            }
            else if (_rotationSequence == "yzx")
            {
                rotationMatrix = rotationX * rotationZ * rotationY;
            }
            else if (_rotationSequence == "zxy")
            {
                rotationMatrix = rotationY * rotationX * rotationZ;
            }
            else if (_rotationSequence == "xzy")
            {
                rotationMatrix = rotationY * rotationZ * rotationX;
            }
            else if (_rotationSequence == "yxz")
            {
                rotationMatrix = rotationZ * rotationX * rotationY;
            }
            else if (_rotationSequence == "xyz")
            {
                rotationMatrix = rotationZ * rotationY * rotationX;
            }

            return rotationMatrix;
        }

        public bool FindChessboardCorners(ref Mat color, out VectorOfPointF chessboardCorners, bool drawChessboardImage = false)
        {
            VectorOfPointF corners = new VectorOfPointF();
            Size patternSize = new Size(8, 5);
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(color, grayImage, ColorConversion.Bgr2Gray);
            if (!CvInvoke.FindChessboardCornersSB(grayImage, patternSize, corners))
            {
                chessboardCorners = corners;
                return false;
            }
            /*VectorOfPointF reverseCornerPoints = new VectorOfPointF();
            for (int i = 0; i < corners.Size; i++)
            {
                PointF temp = corners[corners.Size - 1 - i];
                reverseCornerPoints.Push(new PointF[] { temp });
            }
            corners = reverseCornerPoints;*/

            CvInvoke.CornerSubPix(grayImage, corners, new Size(11, 11), new Size(-1, -1), new MCvTermCriteria(30, 0.1));
            if (drawChessboardImage)
            {
                CvInvoke.DrawChessboardCorners(color, patternSize, corners, drawChessboardImage);
            }

            chessboardCorners = corners;
            return true;
        }

        public static List<PositionRotationData> ReadPositionRotationDataFile(string filepath)
        {
            var dataList = new List<PositionRotationData>();

            using (var reader = new StreamReader(filepath))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    var values = line.Split(',');
                    if (values.Length == 6)
                    {
                        dataList.Add(new PositionRotationData(
                            double.Parse(values[0], CultureInfo.InvariantCulture),
                            double.Parse(values[1], CultureInfo.InvariantCulture),
                            double.Parse(values[2], CultureInfo.InvariantCulture),
                            double.Parse(values[3], CultureInfo.InvariantCulture),
                            double.Parse(values[4], CultureInfo.InvariantCulture),
                            double.Parse(values[5], CultureInfo.InvariantCulture)));
                    }
                }
            }

            return dataList;
        }

        public void CalibrateRobotHandEye(HandEyeCalibrationMethod handEyeCalibrationMethod)
        {
            string[] chessboardImagePath = Directory.GetFiles(_chessboardImageFolderPath);
            if (chessboardImagePath.Length == 0)
            {
                return;
            }
            DebugMessage = "";
            VectorOfVectorOfPointF imagePoints = new VectorOfVectorOfPointF();
            VectorOfVectorOfPoint3D32F objectPoints = new VectorOfVectorOfPoint3D32F();
            Size imageSize = new Size();
            for (int i = 0; i < chessboardImagePath.Length; i++)
            {
                VectorOfPointF chessboardCorners = new VectorOfPointF();
                Mat chessboardImage = CvInvoke.Imread(chessboardImagePath[i]);
                imageSize = chessboardImage.Size;
                if (FindChessboardCorners(ref chessboardImage, out chessboardCorners, true))
                {
                    imagePoints.Push(chessboardCorners);
                    objectPoints.Push(_objectPoints);
                    CvInvoke.Imwrite(Path.Combine(_drawChessboardFolderPath, Path.GetFileName(chessboardImagePath[i])), chessboardImage);
                }
            }
            Mat cameraMatrix = new Mat(3, 3, DepthType.Cv32F, 1);
            cameraMatrix.SetTo(new MCvScalar(0));
            Mat distCoeffs = new Mat(1, 5, DepthType.Cv32F, 1);
            distCoeffs.SetTo(new MCvScalar(0));
            VectorOfMat rvecs = new VectorOfMat();
            VectorOfMat tvecs = new VectorOfMat();
            CvInvoke.CalibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs, CalibType.Default, new MCvTermCriteria(30, 0.1));
            DebugMessage += $"Camera Matrix\n" +
                $"{cameraMatrix.GetValue(0, 0):F4} {cameraMatrix.GetValue(0, 1):F4} {cameraMatrix.GetValue(0, 2):F4}\n" +
                $"{cameraMatrix.GetValue(1, 0):F4} {cameraMatrix.GetValue(1, 1):F4} {cameraMatrix.GetValue(1, 2):F4}\n" +
                $"{cameraMatrix.GetValue(2, 0):F4} {cameraMatrix.GetValue(2, 1):F4} {cameraMatrix.GetValue(2, 2):F4}\n";
            DebugMessage += $"Distortion Coefficient\n" +
                $"{distCoeffs.GetValue(0, 0):F4} {distCoeffs.GetValue(0, 1):F4} {distCoeffs.GetValue(0, 2):F4} {distCoeffs.GetValue(0, 3):F4} {distCoeffs.GetValue(0, 4):F4}\n";


            VectorOfMat tTarget2Cam = new VectorOfMat();
            VectorOfMat rTarget2Cam = new VectorOfMat();
            CameraPoseTransformationMatrix = new List<Matrix<double>>();
            for (int i = 0; i < rvecs.Size; i++)
            {
                tTarget2Cam.Push(tvecs[i]);
                Mat rotationVector = new Mat();
                CvInvoke.Rodrigues(rvecs[i], rotationVector);
                rTarget2Cam.Push(rotationVector);
                DebugMessage += $"\nimage_{i}\n" +
                    $"Camera Translation Vector\n" +
                    $"{tvecs[i].GetValue(0, 0):F4} {tvecs[i].GetValue(0, 1):F4} {tvecs[i].GetValue(0, 2):F4}\n";
                DebugMessage += $"Camera Rotation vector\n" +
                    $"{rotationVector.GetValue(0, 0):F4} {rotationVector.GetValue(0, 1):F4} {rotationVector.GetValue(0, 2):F4}\n" +
                    $"{rotationVector.GetValue(1, 0):F4} {rotationVector.GetValue(1, 1):F4} {rotationVector.GetValue(1, 2):F4}\n" +
                    $"{rotationVector.GetValue(2, 0):F4} {rotationVector.GetValue(2, 1):F4} {rotationVector.GetValue(2, 2):F4}\n";
                Matrix<double> cameraPoseMatrix = new Matrix<double>(4, 4);
                cameraPoseMatrix[0, 0] = rotationVector.GetValue(0, 0); cameraPoseMatrix[0, 1] = rotationVector.GetValue(0, 1); cameraPoseMatrix[0, 2] = rotationVector.GetValue(0, 2); cameraPoseMatrix[0, 3] = tvecs[i].GetValue(0, 0);
                cameraPoseMatrix[1, 0] = rotationVector.GetValue(1, 0); cameraPoseMatrix[1, 1] = rotationVector.GetValue(1, 1); cameraPoseMatrix[1, 2] = rotationVector.GetValue(1, 2); cameraPoseMatrix[1, 3] = tvecs[i].GetValue(0, 1);
                cameraPoseMatrix[2, 0] = rotationVector.GetValue(2, 0); cameraPoseMatrix[2, 1] = rotationVector.GetValue(2, 1); cameraPoseMatrix[2, 2] = rotationVector.GetValue(2, 2); cameraPoseMatrix[2, 3] = tvecs[i].GetValue(0, 2);
                cameraPoseMatrix[3, 0] = 0; cameraPoseMatrix[3, 1] = 0; cameraPoseMatrix[3, 2] = 0; cameraPoseMatrix[3, 3] = 1;
                CameraPoseTransformationMatrix.Add(cameraPoseMatrix);

            }

            VectorOfMat tGripper2Base = new VectorOfMat();
            VectorOfMat rGripper2Base = new VectorOfMat();
            List<PositionRotationData> robotPositionRotationList = ReadPositionRotationDataFile(_robotDataPath);
            RobotPoseTransformationMatrix = new List<Matrix<double>>();
            for (int i = 0; i < robotPositionRotationList.Count; i++)
            {
                var data = robotPositionRotationList[i];
                Mat translationVector = new Mat(1, 3, DepthType.Cv64F, 1);
                translationVector.SetValue(0, 0, data.X); translationVector.SetValue(0, 1, data.Y); translationVector.SetValue(0, 2, data.Z);
                Matrix<double> rotationVector = EulerAngleToRotationmatrix(data.Rx, data.Ry, data.Rz);
                if (IsRotationMatrix(rotationVector))
                {
                    DebugMessage += $"\nimage_{i}\n" +
                        $"Robot Translation Vector\n" +
                        $"{translationVector.GetValue(0, 0):F4} {translationVector.GetValue(0, 1):F4} {translationVector.GetValue(0, 2):F4}\n";
                    DebugMessage += $"Robot Rotation Vector\n" +
                        $"{rotationVector[0, 0]:F4} {rotationVector[0, 1]:F4} {rotationVector[0, 2]:F4}\n" +
                        $"{rotationVector[1, 0]:F4} {rotationVector[1, 1]:F4} {rotationVector[1, 2]:F4}\n" +
                        $"{rotationVector[2, 0]:F4} {rotationVector[2, 1]:F4} {rotationVector[2, 2]:F4}\n";
                    tGripper2Base.Push(translationVector);
                    rGripper2Base.Push(rotationVector.Mat);
                    Matrix<double> robotPoseMatrix = new Matrix<double>(4, 4);
                    robotPoseMatrix[0, 0] = rotationVector[0, 0]; robotPoseMatrix[0, 1] = rotationVector[0, 1]; robotPoseMatrix[0, 2] = rotationVector[0, 2]; robotPoseMatrix[0, 3] = translationVector.GetValue(0, 0);
                    robotPoseMatrix[1, 0] = rotationVector[1, 0]; robotPoseMatrix[1, 1] = rotationVector[1, 1]; robotPoseMatrix[1, 2] = rotationVector[1, 2]; robotPoseMatrix[1, 3] = translationVector.GetValue(0, 1);
                    robotPoseMatrix[2, 0] = rotationVector[2, 0]; robotPoseMatrix[2, 1] = rotationVector[2, 1]; robotPoseMatrix[2, 2] = rotationVector[2, 2]; robotPoseMatrix[2, 3] = translationVector.GetValue(0, 2);
                    robotPoseMatrix[3, 0] = 0; robotPoseMatrix[3, 1] = 0; robotPoseMatrix[3, 2] = 0; robotPoseMatrix[3, 3] = 1;

                    RobotPoseTransformationMatrix.Add(robotPoseMatrix);
                }

            }

            if (tGripper2Base.Size == tTarget2Cam.Size)
            {
                Mat tCam2Gripper = new Mat();
                Mat rCam2Gripper = new Mat();
                CvInvoke.CalibrateHandEye(rGripper2Base, tGripper2Base, rTarget2Cam, tTarget2Cam, rCam2Gripper, tCam2Gripper, handEyeCalibrationMethod);

                string handEyeCalibrationMethodName = "";
                if (handEyeCalibrationMethod == HandEyeCalibrationMethod.Tsai)
                {
                    DebugMessage += $"\nTsai Method\n";
                    handEyeCalibrationMethodName = "Tsai";
                }
                else if (handEyeCalibrationMethod == HandEyeCalibrationMethod.Park)
                {
                    DebugMessage += $"\nPark Method\n";
                    handEyeCalibrationMethodName = "Park";
                }
                else if (handEyeCalibrationMethod == HandEyeCalibrationMethod.Horaud)
                {
                    DebugMessage += $"\nHoraud Method\n";
                    handEyeCalibrationMethodName = "Horaud";
                }
                else if (handEyeCalibrationMethod == HandEyeCalibrationMethod.Andreff)
                {
                    DebugMessage += $"\nAndreff Method\n";
                    handEyeCalibrationMethodName = "Andreff";
                }
                else if (handEyeCalibrationMethod == HandEyeCalibrationMethod.Daniilidis)
                {
                    DebugMessage += $"\nDaniilidis Method\n";
                    handEyeCalibrationMethodName = "Daniilidis";
                }
                DebugMessage +=
                    $"{rCam2Gripper.GetValue(0, 0):F4} {rCam2Gripper.GetValue(0, 1):F4} {rCam2Gripper.GetValue(0, 2):F4} {tCam2Gripper.GetValue(0, 0):F4}\n" +
                    $"{rCam2Gripper.GetValue(1, 0):F4} {rCam2Gripper.GetValue(1, 1):F4} {rCam2Gripper.GetValue(1, 2):F4} {tCam2Gripper.GetValue(1, 0):F4}\n" +
                    $"{rCam2Gripper.GetValue(2, 0):F4} {rCam2Gripper.GetValue(2, 1):F4} {rCam2Gripper.GetValue(2, 2):F4} {tCam2Gripper.GetValue(2, 0):F4}\n";

                DirectoryInfo calibrationDataDirectoryInfo = new DirectoryInfo(_chessboardImageFolderPath);
                string storeCalibrationProcessMessagePath = Path.Combine(calibrationDataDirectoryInfo.Parent.Name, $"CalibrationProcessMessage_{handEyeCalibrationMethodName}.txt");
                File.WriteAllText(storeCalibrationProcessMessagePath, DebugMessage);

                TransformationMatrixX = new Matrix<double>(4, 4);
                TransformationMatrixX[0, 0] = rCam2Gripper.GetValue(0, 0); TransformationMatrixX[0, 1] = rCam2Gripper.GetValue(0, 1); TransformationMatrixX[0, 2] = rCam2Gripper.GetValue(0, 2); TransformationMatrixX[0, 3] = tCam2Gripper.GetValue(0, 0);
                TransformationMatrixX[1, 0] = rCam2Gripper.GetValue(1, 0); TransformationMatrixX[1, 1] = rCam2Gripper.GetValue(1, 1); TransformationMatrixX[1, 2] = rCam2Gripper.GetValue(1, 2); TransformationMatrixX[1, 3] = tCam2Gripper.GetValue(1, 0);
                TransformationMatrixX[2, 0] = rCam2Gripper.GetValue(2, 0); TransformationMatrixX[2, 1] = rCam2Gripper.GetValue(2, 1); TransformationMatrixX[2, 2] = rCam2Gripper.GetValue(2, 2); TransformationMatrixX[2, 3] = tCam2Gripper.GetValue(2, 0);
                TransformationMatrixX[3, 0] = 0; TransformationMatrixX[3, 1] = 0; TransformationMatrixX[3, 2] = 0; TransformationMatrixX[3, 3] = 1;


                DebugMessage += "Calibration Success\n";
            }
            else
            {
                DebugMessage += "Calibration Failed\n";
            }
        }
    }

    internal class Program
    {
        
        static void Main(string[] args)
        {
            string _chessboardFolderPath = "CalibrationData\\Image";
            string _robotDataFolderPath = "CalibrationData\\RobotData";
            string _drawChessboardFolderPath = "CalibrationData\\DrawChessboard";
            CalibrateHandEye calibrateHandEye = new CalibrateHandEye(_chessboardFolderPath, Path.Combine(_robotDataFolderPath, "robotData.txt"), _drawChessboardFolderPath, 8, 5, 30, "xyz");
            calibrateHandEye.CalibrateRobotHandEye(HandEyeCalibrationMethod.Tsai);
            Console.WriteLine(calibrateHandEye.DebugMessage);
            calibrateHandEye.CalibrateRobotHandEye(HandEyeCalibrationMethod.Park);
            Console.WriteLine(calibrateHandEye.DebugMessage);
            calibrateHandEye.CalibrateRobotHandEye(HandEyeCalibrationMethod.Horaud);
            Console.WriteLine(calibrateHandEye.DebugMessage);
            calibrateHandEye.CalibrateRobotHandEye(HandEyeCalibrationMethod.Andreff);
            Console.WriteLine(calibrateHandEye.DebugMessage);
            calibrateHandEye.CalibrateRobotHandEye(HandEyeCalibrationMethod.Daniilidis);
            Console.WriteLine(calibrateHandEye.DebugMessage);
            Console.ReadLine();

        }
    }
}
