// PLY → .ksplat converter (mkkellogg native format).
// compressionLevel 1 = 16-bit half-float quantisation — visually lossless,
// ~half the size, and parses far faster in the browser than text/float PLY.
import * as GS from './gs3d.module.js';
import * as fs from 'fs';

const input  = process.argv[2];
const output = process.argv[3];
const compressionLevel       = process.argv[4] !== undefined ? parseInt(process.argv[4]) : 1;
const alphaRemovalThreshold  = process.argv[5] !== undefined ? parseInt(process.argv[5]) : 1;
const shDegree               = process.argv[6] !== undefined ? parseInt(process.argv[6]) : 2;
const sectionSize = 0, blockSize = 5.0, bucketSize = 256, sceneCenter = undefined;

const buf = fs.readFileSync(input);
const splatArray = GS.PlyParser.parseToUncompressedSplatArray(buf.buffer, shDegree);
const gen = GS.SplatBufferGenerator.getStandardGenerator(
  alphaRemovalThreshold, compressionLevel, sectionSize, sceneCenter, blockSize, bucketSize);
const splatBuffer = gen.generateFromUncompressedSplatArray(splatArray);
fs.writeFileSync(output, Buffer.from(splatBuffer.bufferData));
console.log(`wrote ${output}  (${(fs.statSync(output).size/1e6).toFixed(1)} MB, ` +
            `compression=${compressionLevel}, SH=${shDegree})`);
