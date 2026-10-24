#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

//g++ Sobel.cpp -o Sobel $(pkg-config --cflags --libs opencv4) && ./Sobel

struct Metrics {
    int contours_detectes = 0;
    int contours_reference = 0;
    int contours_corrects = 0;
    int faux_positifs = 0;
    int faux_negatifs = 0;
    double performance = 0.0;
    double tfp = 0.0;
    double tfn = 0.0;
};

Metrics evaluerContours(const cv::Mat& detected, const cv::Mat& reference) {
    Metrics m;
    
    cv::Mat det_bin, ref_bin; 
    // Pour l'image détectée (contours nets) : seuil standard à 127 
    cv::threshold(detected, det_bin, 127, 255, cv::THRESH_BINARY);
    
    // Pour l'image de référence (traits gris clairs) : seuil haut à 230 pour ne pas rater les traits
    cv::threshold(reference, ref_bin, 230, 255, cv::THRESH_BINARY);

    // S'assurer que les deux images ont exactement la même taille
    if (det_bin.size() != ref_bin.size()) {
        cv::resize(ref_bin, ref_bin, det_bin.size(), 0, 0, cv::INTER_NEAREST);
    }

    int rows = det_bin.rows;
    int cols = det_bin.cols;
    int total_pixels = rows * cols;

    // Total détectés : Pixels sombres (< 127) dans l'image détectée
    m.contours_detectes = total_pixels - cv::countNonZero(det_bin > 127);

    // Total référence : Pixels sombres (< 127) dans l'image de référence
    m.contours_reference = total_pixels - cv::countNonZero(ref_bin > 127);

    // Calcul des contours corrects avec voisinage 3x3 
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            if (det_bin.at<uchar>(y, x) < 127) { // Si pixel sombre détecté
                bool found_in_neighborhood = false;
                
                // Explorer le voisinage 3x3 (-1, 0, +1)
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int ny = y + dy;
                        int nx = x + dx;
                        if (ny >= 0 && ny < rows && nx >= 0 && nx < cols) {
                            if (ref_bin.at<uchar>(ny, nx) < 127) { // Si référence a un contour proche
                                found_in_neighborhood = true;
                                break;
                            }
                        }
                    }
                    if (found_in_neighborhood) break;
                }
                
                if (found_in_neighborhood) {
                    m.contours_corrects++;
                }
            }
        }
    }

    m.faux_positifs = m.contours_detectes - m.contours_corrects;
    m.faux_negatifs = m.contours_reference - m.contours_corrects;

    // Calcul des mesures P, TFP, TFN 
    double total = (double)(m.contours_corrects + m.faux_positifs + m.faux_negatifs);
    if (total > 0) {
        m.performance = (double)m.contours_corrects / total;
        m.tfp = (double)m.faux_positifs / total;
        m.tfn = (double)m.faux_negatifs / total;
    }

    return m;
}

// Variables globales pour que la fonction de la souris puisse y accéder
cv::Mat contours;
cv::Mat reference;

// Fonction appelée automatiquement à chaque clic de souris
void mouseCallback(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) { // Si clic gauche de la souris
        if (contours.empty() || reference.empty()) {
            std::cout << "\n[Attention] Image de reference introuvable ou contours vides " << std::endl;
            return;
        }

        Metrics m = evaluerContours(contours, reference);

        std::cout << "\n================================" << std::endl;
        std::cout << "      RAPPORT D'EVALUATION        " << std::endl;
        std::cout << "================================" << std::endl;
        std::cout << "- Contours detectes : " << m.contours_detectes << std::endl;
        std::cout << "- Contours reference: " << m.contours_reference << std::endl;
        std::cout << "- Contours corrects : " << m.contours_corrects << std::endl;
        std::cout << "- Faux positifs     : " << m.faux_positifs << std::endl;
        std::cout << "- Faux negatifs     : " << m.faux_negatifs << std::endl;
        std::cout << "--------------------------------" << std::endl;
        std::cout << "- Performance (P)   : " << m.performance << std::endl;
        std::cout << "- Taux Faux Positifs: " << m.tfp << std::endl;
        std::cout << "- Taux Faux Negatifs: " << m.tfn << std::endl;
        std::cout << "================================\n" << std::endl;
    }
}


int main() {
    // Comment fonctionne le programme :
        std::cout << "\n================================================================" << std::endl;
        std::cout << "                             TP CONTOURS                          " << std::endl;
        std::cout << "================================================================" << std::endl;
        std::cout << "Pour parcourir le lot d'images, cliquez sur les flêches de votre clavier," << std::endl;
        std::cout << "sinon, cliquez sur les touches 'd' et 'g' de votre clavier." << std::endl;
        std::cout << "Utilisez les trackbars pour ajuster les 4 paramètres suivants:" << std::endl;
        std::cout << "- Detecteur (0:Sobel, 1:Laplace, 2:Canny)" << std::endl;
        std::cout << "- Taille du filtre (3,5 ou 7)" << std::endl;
        std::cout << "- Param 1 (Seuil pour Sobel et Laplace + seuil bas pour canny)" << std::endl;
        std::cout << "- Param 2 (Seuil haut pour Canny) \n\n" << std::endl;

    // Récupérer toutes les images du dossier "images" et les trier par ordre alphabétique
    std::vector<std::string> image_paths;
    cv::glob("images/*.jpg", image_paths); 
    std::sort(image_paths.begin(), image_paths.end());
    if (image_paths.empty()) {
        std::cout << "Aucune image trouvee dans le dossier images !" << std::endl;
        return -1;
    }


    //Recuperer les references
     std::vector<std::string> ref_paths;
    cv::glob("references/*.jpg", ref_paths); 
    std::sort(ref_paths.begin(), ref_paths.end());
    if (ref_paths.empty()) {
        std::cout << "Aucune reference trouvee dans le dossier references !" << std::endl;
        return -1;
    }
    size_t img_index = 0; // Index de l'image actuelle

    // Variables pour les trackbars
    int detector_type = 0; // 0: Sobel, 1: Laplace, 2: Canny
    int filter_size = 0;   // Taille du filtre (3, 5 ou 7)
    int param1 = 80;       // Seuil 1 de détection pour Sobel
    int param2 = 150;      // Seuil 2 pour Canny

    
    // Création de la fenêtre et des trackbars propres 
    cv::namedWindow("Contours Evaluation");
    cv::createTrackbar("Detecteur (0:Sobel, 1:Laplace, 2:Canny)", "Contours Evaluation", &detector_type, 2);
    cv::createTrackbar("Taille du filtre (3,5 ou 7)", "Contours Evaluation", &filter_size, 2); 
    cv::createTrackbar("Param 1 (Seuil pour Sobel et Laplace + seuil bas pour canny)", "Contours Evaluation", &param1, 255); 
    cv::createTrackbar("Param 2 (Seuil haut pour Canny)", "Contours Evaluation", &param2, 255); 



    // Attacher la fonction souris à la fenêtre
    cv::setMouseCallback("Contours Evaluation", mouseCallback);

    while(true) {

        // Charger l'image courante à chaque tour de boucle
        cv::Mat image = cv::imread(image_paths[img_index], cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cout << "Impossible de charger l'image : " << image_paths[img_index] << std::endl;
            break;
        }

        // Charger la reference de l'image courante à chaque tour de boucle
        reference = cv::imread(ref_paths[img_index], cv::IMREAD_GRAYSCALE);
        if (reference.empty()) {
            std::cout << "Impossible de charger la reference : " << ref_paths[img_index] << std::endl;
            break;
        }


        //determiner taille du filtre
        int kernel_size = 3; // Valeur par défaut
        if (filter_size == 0) kernel_size = 3;
        else if (filter_size == 1) kernel_size = 5;
        else if (filter_size == 2) kernel_size = 7;

        if (detector_type == 0) { // Sobel
            cv::Mat grad_x, grad_y;
            cv::Sobel(image, grad_x, CV_16S, 1, 0, kernel_size);
            cv::Sobel(image, grad_y, CV_16S, 0, 1, kernel_size);

            cv::Mat abs_grad_x, abs_grad_y;
            cv::convertScaleAbs(grad_x, abs_grad_x);
            cv::convertScaleAbs(grad_y, abs_grad_y);

            cv::Mat sobel;
            cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);
            
            cv::threshold(sobel, contours, param1, 255, cv::THRESH_BINARY);
            cv::bitwise_not(contours, contours);

        }else if (detector_type == 1) { // Laplace
            cv::Mat laplacian, abs_laplacian;
            cv::Laplacian(image, laplacian, CV_16S, kernel_size); // Le 'kernel_size' est la taille du kernel (voisinage 3x3)
            cv::convertScaleAbs(laplacian, contours);
            cv::threshold(contours, contours, param1, 255, cv::THRESH_BINARY);
            cv::bitwise_not(contours, contours);

        } else { // Canny
            int param2 = 150; // Seuil 2 pour Canny
            cv::Canny(image, contours, param1, param2, kernel_size);
            cv::bitwise_not(contours, contours);
        }
        

        // Coller l'image originale et les contours côte à côte 
        cv::Mat affichage_combine;
        cv::hconcat(image, reference, affichage_combine);
        cv::hconcat(affichage_combine, contours, affichage_combine);


        // Afficher les images une fenêtre unique
        cv::imshow("Contours Evaluation", affichage_combine);


        // Attendre 30ms et vérifier si l'utilisateur appuie sur la touche Échap (27) pour quitter
        char key = (char)cv::waitKey(30);
        if (key == 27) { // 27 = Touche ESC
            break;
        }// Flèche droite (ou touche 'd' par sécurité) pour passer à l'image suivante
        else if (key == 39 || key == 'd' || key == 'Q' || key == 2555904|| key == 	65361) { 
            img_index = (img_index + 1) % image_paths.size();
        }
        // Flèche gauche (ou touche 'q' par sécurité) pour revenir à l'image précédente
        else if (key == 37 || key == 2424832 || key == 'g' || key == 'S' || key == 	65363) { 
            img_index = (img_index == 0) ? image_paths.size() - 1 : img_index - 1;
        }
    }

    cv::destroyAllWindows();
    return 0;
}