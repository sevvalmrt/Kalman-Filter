#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;

// Ortalama ve standart sapma ile rastgele gürültü üretme
float generateNoise(float mean, float stddev) {
    static random_device rd;
    static mt19937 gen(rd());
    normal_distribution<float> dist(mean, stddev);
    return dist(gen);
}

// Sensör verilerini gürültü ve drift ile simüle etmek
Point2f simulateSensor(const Point2f& truePosition, float noiseStd, float drift) {
    return Point2f(truePosition.x + generateNoise(0, noiseStd) + drift,
        truePosition.y + generateNoise(0, noiseStd) + drift);
}

int main() {
    // Simülasyon parametreleri
    const int numSteps = 100;
    const float dt = 0.1f; // zaman adımı (değişken)
    const float processNoiseStd = 0.1f;
    const float measurementNoiseStd1 = 0.2f;
    const float measurementNoiseStd2 = 0.1f;
    const float sensorDrift1 = 0.05f;
    const float sensorDrift2 = 0.03f;

    // Kalman Filtresini başlatma
    KalmanFilter kf(4, 2, 0); // 4 durum değişkeni, 2 ölçüm
    kf.transitionMatrix = (Mat_<float>(4, 4) <<
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1);

    kf.measurementMatrix = (Mat_<float>(2, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0);

    kf.processNoiseCov = (Mat_<float>(4, 4) <<
        processNoiseStd, 0, 0, 0,
        0, processNoiseStd, 0, 0,
        0, 0, processNoiseStd, 0,
        0, 0, 0, processNoiseStd);

    // İlk hata kovaryansı
    kf.errorCovPost = Mat::eye(4, 4, CV_32F);
    kf.statePost = (Mat_<float>(4, 1) << 0, 0, 0, 0);  // Başlangıç durumu

    // Gerçek Veri ve Sensör Ölçümleri
    vector<Point2f> groundTruth;
    vector<Point2f> measurements1, measurements2;
    vector<Point2f> kalmanEstimates;
    vector<float> errors;  // Hataları saklamak için bir vektör

    Point2f currentPosition(0, 0);
    for (int i = 0; i < numSteps; i++) {
        // Gerçek Pozisyonu Güncelleme
        currentPosition.x += 1.0f * dt;
        currentPosition.y += 0.5f * dt;
        groundTruth.push_back(currentPosition);

        // Gürültülü sensör ölçümlerini simüle etmek
        measurements1.push_back(simulateSensor(currentPosition, measurementNoiseStd1, sensorDrift1));
        measurements2.push_back(simulateSensor(currentPosition, measurementNoiseStd2, sensorDrift2));

        // İki sensör ölçümünü ortalamak (basitlik açısından)
        Point2f combinedMeasurement = (measurements1.back() + measurements2.back()) * 0.5;
        Mat measurement = (Mat_<float>(2, 1) << combinedMeasurement.x, combinedMeasurement.y);

        // Kalman filtresi prediction ve correction
        Mat prediction = kf.predict();

        // Ağırlıklı sensör verileri ile düzeltme adımı
        float weight1 = 1.0f / (measurementNoiseStd1 * measurementNoiseStd1);
        float weight2 = 1.0f / (measurementNoiseStd2 * measurementNoiseStd2);

        float totalWeight = weight1 + weight2;
        weight1 /= totalWeight;
        weight2 /= totalWeight;

        // Sensör ölçümlerinin ağırlıklı ortalaması
        Point2f weightedMeasurement = measurements1.back() * weight1 + measurements2.back() * weight2;
        Mat weightedMeasurementMat = (Mat_<float>(2, 1) << weightedMeasurement.x, weightedMeasurement.y);

        // Ağırlıklı ölçümle correction adımını gerçekleştirme
        Mat estimate = kf.correct(weightedMeasurementMat);
        kalmanEstimates.push_back(Point2f(estimate.at<float>(0), estimate.at<float>(1)));

        // Hata oranını hesapla (gerçek pozisyon ile tahmin arasındaki fark)
        float error = sqrt(pow(currentPosition.x - kalmanEstimates.back().x, 2) +
            pow(currentPosition.y - kalmanEstimates.back().y, 2));
        errors.push_back(error);
    }

    // Görselleştirme
    Mat display(600, 800, CV_8UC3, Scalar(255, 255, 255));

    // Gerçek yolun çizilmesi
    for (size_t i = 1; i < groundTruth.size(); i++) {
        int y1 = display.rows - groundTruth[i - 1].y * 100;
        int y2 = display.rows - groundTruth[i].y * 100;
        line(display, Point(groundTruth[i - 1].x * 100, y1),
            Point(groundTruth[i].x * 100, y2), Scalar(0, 255, 0), 2);
    }

    // Sensör 1'in yolunu çizme (mavi noktalar)
    for (size_t i = 0; i < measurements1.size(); i++) {
        int y = display.rows - measurements1[i].y * 100;
        circle(display, Point(measurements1[i].x * 100, y), 3, Scalar(255, 0, 0), -1);
    }

    // Sensör 2'nin yolunu çizme (kırmızı noktalar)
    for (size_t i = 0; i < measurements2.size(); i++) {
        int y = display.rows - measurements2[i].y * 100;
        circle(display, Point(measurements2[i].x * 100, y), 3, Scalar(0, 0, 255), -1);
    }

    // Kalman filtresi tahmini çizme (mor çizgiler)
    for (size_t i = 1; i < kalmanEstimates.size(); i++) {
        int y1 = display.rows - kalmanEstimates[i - 1].y * 100;
        int y2 = display.rows - kalmanEstimates[i].y * 100;
        line(display, Point(kalmanEstimates[i - 1].x * 100, y1),
            Point(kalmanEstimates[i].x * 100, y2), Scalar(255, 0, 255), 2);
    }

    // Son görselleştirme
    imshow("Kalman Filter Visualization", display);

    // Hata oranlarını gösteren pencere
    Mat errorGraph(300, 600, CV_8UC3, Scalar(255, 255, 255));
    for (size_t i = 1; i < errors.size(); i++) {
        line(errorGraph, Point((i - 1) * 6, 300 - errors[i - 1] * 50),
            Point(i * 6, 300 - errors[i] * 50), Scalar(0, 0, 0), 2);
    }
    imshow("Error Graph", errorGraph);

    waitKey(0);

    return 0;
}
