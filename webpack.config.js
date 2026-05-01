const path = require('path');

module.exports = {
  mode: 'none',
  entry: './src/extension.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'extension.js',
    libraryTarget: 'commonjs',
  },
  target: 'node',
  externals: {
    vscode: 'commonjs vscode',
    'onnxruntime-node': 'commonjs onnxruntime-node',
    sharp: 'commonjs sharp'
  },
  resolve: {
    extensions: ['.ts', '.js'],
  },
  module: {
    rules: [
      {
        test: /\.ts$/,
        exclude: /node_modules/,
        use: [
          {
            loader: 'ts-loader',
          },
        ],
      },
    ],
  },
  devtool: 'nosources-source-map',
};
