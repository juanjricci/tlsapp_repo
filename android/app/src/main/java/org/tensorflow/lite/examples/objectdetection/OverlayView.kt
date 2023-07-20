/*
 * Copyright 2022 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import org.tensorflow.lite.task.vision.detector.Detection
import java.util.*
import kotlin.math.max


class OverlayView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {

    private var results: List<Detection> = LinkedList<Detection>()
    private var boxPaint = Paint()
    private var textBackgroundPaint = Paint()
    private var textPaint = Paint()
    private var lowTextPaint = Paint()

    private var scaleFactor: Float = 1f

    private var bounds = Rect()

    private var label = ""
    private var score = 0.1f
    //var words = mutableListOf<String>()

    var letters_per_line = 24

    var detectedSigns = ""
    var second_line = ""
    var third_line = ""

    private var detected = false
    private var lastLetter = ""



    init {
        initPaints()
    }

    fun clear() {
        textPaint.reset()
        textBackgroundPaint.reset()
        boxPaint.reset()
        invalidate()
        initPaints()
    }

    private fun initPaints() {
        textBackgroundPaint.color = Color.BLACK
        textBackgroundPaint.style = Paint.Style.FILL
        textBackgroundPaint.textSize = 100f

        textPaint.color = Color.WHITE
        textPaint.style = Paint.Style.FILL
        textPaint.textSize = 100f

        lowTextPaint.color = Color.WHITE
        lowTextPaint.style = Paint.Style.FILL
        lowTextPaint.textSize = 50f

        boxPaint.color = ContextCompat.getColor(context!!, R.color.bounding_box_color)
        boxPaint.strokeWidth = 12F
        boxPaint.style = Paint.Style.STROKE
    }

    override fun draw(canvas: Canvas) {

        super.draw(canvas)

        // si no hay detecciones...
        if (results.isEmpty()){
            detected = false

        }

        for (result in results) {

            val boundingBox = result.boundingBox

            val top = boundingBox.top * scaleFactor
            val bottom = boundingBox.bottom * scaleFactor
            val left = boundingBox.left * scaleFactor
            val right = boundingBox.right * scaleFactor

            // Dibujo un rectángulo alrededor del gesto
            val drawableRect = RectF(left, top, right, bottom)
            canvas.drawRect(drawableRect, boxPaint)

            // Guardo las detecciones en variables
            label = result.categories[0].label
            score = result.categories[0].score


            // Create text to display alongside detected objects
            val drawableText =
                label + " " +
                        String.format("%.0f%%", score * 100)

            // Dibujo un rectángulo atrás de la label y el score
            textBackgroundPaint.getTextBounds(drawableText, 0, drawableText.length, bounds)
            val textWidth = bounds.width()
            val textHeight = bounds.height()
            canvas.drawRect(
                left,
                top,
                left + textWidth + BOUNDING_RECT_TEXT_PADDING,
                top + textHeight + BOUNDING_RECT_TEXT_PADDING,
                textBackgroundPaint
            )

            // Escribo la label y el score
            canvas.drawText(drawableText, left, top + bounds.height(), textPaint)

            // Tomo la última letra del string q se está formando
            lastLetter = detectedSigns.takeLast(1)
            if (second_line.isNotEmpty()){
                lastLetter = second_line.takeLast(1)
                if (third_line.isNotEmpty()){
                    lastLetter = third_line.takeLast(1)
                }
            }
            // Verifico que la nueva letra sea distinta a la anterior, así no se escriben múltiples veces
            val newSign = label
            if (detected == false){
                if (score > 0.80){
                    if (third_line.length == letters_per_line) {
                        clearAll()
                    }
                    if (newSign == "ESPACIO"){
                        detected = true
                        detectedSigns += " "
                    }
                    else {
                        if (detectedSigns.length >= letters_per_line){
                            if (second_line.length >= letters_per_line){
                                detected = true
                                third_line += newSign
                            }
                            else{
                                detected = true
                                second_line += newSign
                            }
                        }
                        else {
                            detected = true
                            detectedSigns += newSign
                        }
                    }
                }
            } else {
                if (score < 0.50 || (newSign != lastLetter) ) {
                    detected = false
                }
            }
        }

        // Dibujo otro rectángulo al final del canvas
        canvas.drawRect(0f, height - 410f, width.toFloat(), height - 100f, textBackgroundPaint)

        // Escribo en pantalla el string formado con las detecciones
        canvas.drawText(detectedSigns, 50f, height - 300f, lowTextPaint)
        canvas.drawText(second_line, 50f, height - 250f, lowTextPaint)
        canvas.drawText(third_line, 50f, height - 200f, lowTextPaint)

    }

    fun setResults(
      detectionResults: MutableList<Detection>,
      imageHeight: Int,
      imageWidth: Int,
    ) {
        results = detectionResults

        // PreviewView is in FILL_START mode. So we need to scale up the bounding box to match with
        // the size that the captured images will be displayed.
        scaleFactor = max(width * 1f / imageWidth, height * 1f / imageHeight)
    }

    fun clearDetectedSigns() {
        if (second_line.isEmpty()){
            val lastletter = detectedSigns.takeLast(1)
            val wo_last = detectedSigns.removeSuffix(lastletter)
            detectedSigns = wo_last
        }
        else {
            val lastletter = second_line.takeLast(1)
            val wo_last = second_line.removeSuffix(lastletter)
            second_line = wo_last
        }
        invalidate()
    }

    fun clearAll() {
        detectedSigns = ""
        second_line = ""
        third_line = ""
        invalidate()
    }

    companion object {
        private const val BOUNDING_RECT_TEXT_PADDING = 8
    }
}
