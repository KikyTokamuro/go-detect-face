package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"log"
	"os"
	"time"

	pigo "github.com/esimov/pigo/core"
)

type Finder struct {
	face *pigo.Pigo
}

func cropImage(img image.Image, crop image.Rectangle) (image.Image, error) {
	type subImager interface {
		SubImage(r image.Rectangle) image.Image
	}

	simg, ok := img.(subImager)
	if !ok {
		return nil, fmt.Errorf("image does not support cropping")
	}

	return simg.SubImage(crop), nil
}

func getImageFromFilePath(filePath string) (image.Image, error) {
	src, err := pigo.GetImage(filePath)
	if err != nil {
		return nil, err
	}
	return src, nil
}

func writeImagesToFile(images []image.Image) error {
	for i, image := range images {
		f, err := os.Create(fmt.Sprintf("%d.jpg", i))
		if err != nil {
			return err
		}
		err = jpeg.Encode(f, image, nil)
		if err != nil {
			return err
		}
		f.Close()
	}

	return nil
}

func (f *Finder) initFaceDetect() error {
	model, err := os.ReadFile("./facefinder.model")
	if err != nil {
		return fmt.Errorf("failed to load face finder model: %s", err)
	}

	p := pigo.NewPigo()

	f.face, err = p.Unpack(model)
	if err != nil {
		return fmt.Errorf("failed to initialize face classifier: %s", err)
	}
	return nil
}

func (f *Finder) detectFace(img image.Image) []image.Image {
	pixels := pigo.RgbToGrayscale(img)
	cols, rows := img.Bounds().Max.X, img.Bounds().Max.Y

	params := pigo.CascadeParams{
		MinSize:     100,
		MaxSize:     600,
		ShiftFactor: 0.15,
		ScaleFactor: 1.1,
		ImageParams: pigo.ImageParams{
			Pixels: pixels,
			Rows:   rows,
			Cols:   cols,
			Dim:    cols,
		},
	}

	dets := f.face.RunCascade(params, 0.0)
	dets = f.face.ClusterDetections(dets, 0)

	detectedFaces := []image.Image{}

	for _, det := range dets {
		if det.Q >= 5.0 {
			faceImg, err := cropImage(
				img,
				image.Rect(
					int(det.Col-det.Scale/2),
					int(det.Row-det.Scale/2),
					int(det.Scale+det.Col-det.Scale/2),
					int(det.Scale+det.Row-det.Scale/4),
				),
			)
			if err != nil {
				continue
			}
			detectedFaces = append(detectedFaces, faceImg)
		}
	}

	return detectedFaces
}

func main() {
	finder := Finder{}
	err := finder.initFaceDetect()
	if err != nil {
		log.Fatal(err)
	}

	img, err := getImageFromFilePath("./_test.jpg")
	if err != nil {
		log.Fatal(err)
	}

	t := time.Now()

	faces := finder.detectFace(img)
	writeImagesToFile(faces)

	log.Println(time.Since(t))
}
